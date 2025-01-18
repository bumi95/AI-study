import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from training.model import RainbowDQN
import numpy as np
from collections import deque
import random
from training.segment_tree import SumSegmentTree, MinSegmentTree

class PrioritizedReplayBuffer:
    def __init__(self, size, alpha=0.6):
        self.alpha = alpha
        self.size = size
        self.next_idx = 0
        self.size_now = 0
        
        self.memory = []
        self.priorities = SumSegmentTree(size)
        self.min_priorities = MinSegmentTree(size)
        self.max_priority = 1.0
        
    def add(self, state, action, reward, next_state, done):
        if len(self.memory) < self.size:
            self.memory.append(None)
        self.memory[self.next_idx] = (state, action, reward, next_state, done)
        
        priority = self.max_priority ** self.alpha
        self.priorities[self.next_idx] = priority
        self.min_priorities[self.next_idx] = priority
        
        self.next_idx = (self.next_idx + 1) % self.size
        self.size_now = min(self.size, self.size_now + 1)
        
    def sample(self, batch_size, beta=0.4):
        indices = []
        weights = np.zeros(batch_size, dtype=np.float32)
        total = self.priorities.sum(0, self.size_now)
        
        min_priority = self.min_priorities.min(0, self.size_now)
        max_weight = (min_priority / total * self.size_now) ** (-beta)
        
        for i in range(batch_size):
            mass = random.random() * total
            idx = self.priorities.find_prefixsum_idx(mass)
            indices.append(idx)
            
            p_sample = self.priorities[idx] / total
            weight = (p_sample * self.size_now) ** (-beta)
            weights[i] = weight / max_weight
            
        samples = [self.memory[idx] for idx in indices]
        return samples, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            priority = (priority + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.min_priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

class RainbowAgent:
    def __init__(self, state_size, action_size, lr=1e-4, gamma=0.99, batch_size=32, n_step=3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.n_step = n_step
        
        # Networks
        self.policy_net = RainbowDQN(state_size, action_size=action_size).to(self.device)
        self.target_net = RainbowDQN(state_size, action_size=action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = PrioritizedReplayBuffer(100000)
        self.n_step_buffer = deque(maxlen=n_step)
        
        # Distributional DQN parameters
        self.num_atoms = 51
        self.v_min = -10
        self.v_max = 10
        self.supports = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(self.device)
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        
    def _get_n_step_info(self):
        reward, next_state, done = self.n_step_buffer[-1][-3:]
        
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_s, d = transition[-3:]
            reward = r + self.gamma * reward * (1 - d)
            next_state, done = (n_s, d) if d else (next_state, done)
            
        return reward, next_state, done
        
    def remember(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) == self.n_step:
            reward, next_state, done = self._get_n_step_info()
            state, action = self.n_step_buffer[0][:2]
            self.memory.add(state, action, reward, next_state, done)
    
    def train_step(self):
        if self.memory.size_now < self.batch_size:
            return
        
        batch, indices, weights = self.memory.sample(self.batch_size, beta=0.4)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Compute current distribution
        current_dist = self.policy_net(states)
        current_dist = current_dist[range(self.batch_size), actions]
        
        # Compute target distribution
        with torch.no_grad():
            next_dist = self.target_net(next_states)
            next_actions = (next_dist * self.supports.expand_as(next_dist)).sum(2).max(1)[1]
            next_dist = next_dist[range(self.batch_size), next_actions]
            
            # Compute projected distribution
            projected_dist = self._categorical_projection(next_dist, rewards, dones)
        
        # Compute loss with importance sampling weights
        elementwise_loss = -(projected_dist * current_dist.log()).sum(1)
        loss = (elementwise_loss * weights).mean()
        
        # Update priorities
        priorities = elementwise_loss.detach().cpu().numpy()
        self.memory.update_priorities(indices, priorities)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()
        
        # Reset noisy layers
        self.policy_net.reset_noise()
        self.target_net.reset_noise()
        
        return loss.item()
    
    def _categorical_projection(self, next_dist, rewards, dones):
        batch_size = next_dist.size(0)
        projected_dist = torch.zeros(batch_size, self.num_atoms).to(self.device)
        
        # Compute projected atoms
        for atom in range(self.num_atoms):
            tz = rewards + (1 - dones) * self.gamma * self.supports[atom]
            tz = tz.clamp(self.v_min, self.v_max)
            b = (tz - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()
            
            # Distribute probability
            projected_dist.scatter_add_(1, l, next_dist[:, atom] * (u.float() - b))
            projected_dist.scatter_add_(1, u, next_dist[:, atom] * (b - l.float()))
        
        return projected_dist
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict()) 