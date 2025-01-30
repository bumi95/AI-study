import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from training.model import RainbowDQN
import numpy as np
from collections import deque
import random
from training.segment_tree import SumSegmentTree, MinSegmentTree
import os

class PrioritizedReplayBuffer:
    def __init__(self, size, alpha=0.6):
        self.alpha = alpha
        self.size = size
        self.next_idx = 0
        self.size_now = 0
        
        # 메모리 버퍼 초기화
        self.memory = []
        # segment tree 크기를 2의 거듭제곱으로 조정
        tree_capacity = 1
        while tree_capacity < size:
            tree_capacity *= 2
            
        self.priorities = SumSegmentTree(tree_capacity)
        self.min_priorities = MinSegmentTree(tree_capacity)
        self.max_priority = 1.0
    
    def add(self, state, action, reward, next_state, done):
        """경험 추가"""
        if len(self.memory) < self.size:
            self.memory.append(None)
        
        # 메모리에 경험 저장
        self.memory[self.next_idx] = (state, action, reward, next_state, done)
        
        # 우선순위 업데이트
        priority = self.max_priority ** self.alpha
        self.priorities[self.next_idx] = priority
        self.min_priorities[self.next_idx] = priority
        
        # 인덱스 업데이트
        self.next_idx = (self.next_idx + 1) % self.size
        self.size_now = min(self.size, self.size_now + 1)
    
    def sample(self, batch_size, beta=0.4):
        """배치 샘플링"""
        if self.size_now == 0:
            return None, None, None
            
        indices = []
        weights = np.zeros(batch_size, dtype=np.float32)
        total = self.priorities.sum(0, self.size_now)
        
        if total <= 0:
            total = 1e-8
            
        min_priority = self.min_priorities.min(0, self.size_now)
        if min_priority <= 0:
            min_priority = 1e-8
            
        max_weight = (min_priority / total * self.size_now) ** (-beta)
        
        for i in range(batch_size):
            mass = random.random() * total
            idx = self.priorities.find_prefixsum_idx(mass)
            if idx >= self.size_now:  # 인덱스 범위 체크
                idx = random.randint(0, max(0, self.size_now - 1))
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
    def __init__(self, state_size, action_size, lr=1e-5, gamma=0.95, batch_size=64, n_step=1):
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
        self.memory = PrioritizedReplayBuffer(500000)
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
        """한 스텝의 학습을 수행"""        
        try:
            # 배치 샘플링
            batch = self.memory.sample(self.batch_size, beta=0.4)
            if batch[0] is None:
                return None
            
            samples, indices, weights = batch
            states, actions, rewards, next_states, dones = zip(*samples)
            
            # numpy 배열을 PyTorch 텐서로 변환
            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)
            weights = torch.FloatTensor(weights).to(self.device)
            
            # NaN 체크
            if torch.isnan(states).any() or torch.isnan(rewards).any():
                print("Warning: NaN detected in input data")
                return None
            
            # 현재 상태의 가치 분포 계산
            current_dist = self.policy_net(states)
            current_dist = current_dist[range(self.batch_size), actions]
            
            # 로그 계산 시 안정성을 위해 작은 값 추가
            current_dist = current_dist.clamp(min=1e-7)
            
            # 다음 상태의 가치 분포 계산
            with torch.no_grad():
                next_dist = self.target_net(next_states)
                next_actions = (next_dist * self.supports.expand_as(next_dist)).sum(2).max(1)[1]
                next_dist = next_dist[range(self.batch_size), next_actions]
                
                # Categorical DQN의 분포 투영
                projected_dist = self._categorical_projection(next_dist, rewards, dones)
            
            # Cross-entropy 손실 계산
            elementwise_loss = -(projected_dist * current_dist.log()).sum(1)
            
            # NaN이나 무한대 체크
            if torch.isnan(elementwise_loss).any() or torch.isinf(elementwise_loss).any():
                print("Warning: NaN or Inf detected in loss calculation")
                return None
            
            loss = (elementwise_loss * weights).mean()
            
            # 최종 loss가 유효한지 확인
            if torch.isnan(loss) or torch.isinf(loss):
                print("Warning: Final loss is NaN or Inf")
                return None
            
            # 그래디언트 계산 및 클리핑
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)  # 그래디언트 클리핑 강화
            
            # 그래디언트가 유효한지 확인
            for param in self.policy_net.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print("Warning: NaN or Inf detected in gradients")
                        return None
            
            self.optimizer.step()
            
            # 우선순위 업데이트
            priorities = elementwise_loss.detach().cpu().numpy()
            self.memory.update_priorities(indices, priorities)
            
            # Noisy 레이어 리셋
            self.policy_net.reset_noise()
            self.target_net.reset_noise()
            
            return loss.item()
            
        except Exception as e:
            print(f"Error in train_step: {e}")
            return None
    
    def _categorical_projection(self, next_dist, rewards, dones):
        """Categorical 알고리즘의 분포 투영"""
        batch_size = len(rewards)
        
        # rewards와 dones가 이미 텐서인지 확인하고 적절히 변환
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.FloatTensor(rewards).to(self.device)
        if not isinstance(dones, torch.Tensor):
            dones = torch.FloatTensor(dones).to(self.device)
        
        # 델타 z 계산
        delta_z = float(self.v_max - self.v_min) / (self.num_atoms - 1)
        
        # 투영된 값 계산
        projected_dist = torch.zeros(batch_size, self.num_atoms).to(self.device)
        
        for atom in range(self.num_atoms):
            tz_j = rewards + (1 - dones) * self.gamma * (self.v_min + atom * delta_z)
            tz_j = tz_j.clamp(self.v_min, self.v_max)
            b = (tz_j - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()
            
            # 인덱스 텐서의 차원을 맞춤
            l_idx = (l.unsqueeze(1) == torch.arange(self.num_atoms).to(self.device)).float()
            u_idx = (u.unsqueeze(1) == torch.arange(self.num_atoms).to(self.device)).float()
            
            projected_dist += l_idx * next_dist * (u.float().unsqueeze(1) - b.unsqueeze(1))
            projected_dist += u_idx * next_dist * (b.unsqueeze(1) - l.float().unsqueeze(1))
        
        return projected_dist
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict()) 

    def save_model(self, path='models/rainbow_model.pth'):
        """모델 저장"""
        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 모델 상태 저장
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'memory': self.memory,  # 리플레이 버퍼도 저장
        }, path)
        print(f"모델이 저장되었습니다: {path}")

    def load_model(self, path='models/rainbow_model.pth'):
        """모델 로드"""
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.memory = checkpoint['memory']
            print(f"모델을 로드했습니다: {path}")
            return True
        return False 