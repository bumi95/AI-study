import torch
import torch.optim as optim
from model import A3CModel
import numpy as np
import logging
import os
import copy
import random

logging.basicConfig(level=logging.INFO)

class A3CAgent:
    def __init__(self, input_size, action_size, lr=0.0001):
        self.input_size = input_size
        self.action_size = action_size
        self.model = A3CModel(input_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        #self.epsilon = 1.0
        #self.epsilon_decay = 0.995
        #self.epsilon_min = 0.1
        #self.gamma = 0.99
        
    def get_action(self, state):
        #state = torch.from_numpy(state).float().unsqueeze(0)
        
        #if np.random.rand() <= self.epsilon:
        #    return random.randrange(self.action_size)
        #else:
        #    policy, _ = self.model(state)
        #    if torch.any(torch.isnan(policy)) or torch.any(policy < 0) or torch.any(torch.isinf(policy)):
        #        policy = torch.softmax(policy, dim=-1)
                
        #    action = torch.multinomial(policy, 1).item()
        #    return action
        state = torch.from_numpy(state).float()#.unsqueeze(0)
        policy, value = self.model(state)
        #action = torch.multinomial(policy, 1).item()
        return policy, value#action, value

    
    def update(self, trajectory, global_model, total_reward, record):
        #states = torch.tensor(np.array([x[0] for x in trajectory]), dtype=torch.float32)
        log_probs = [x[0] for x in trajectory]
        values = torch.cat([x[1] for x in trajectory])
        rewards = [x[2] for x in trajectory]
        
        # Discounted rewards
        discounted_rewards = []
        R = 0
        for reward in reversed(rewards):
            R = reward + 0.99 * R
            discounted_rewards.insert(0, R)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        
        # Compute loss
        #policy, values = self.model(states)
        advantage = discounted_rewards - values.squeeze()
        
        #log_probs = torch.log(policy.gather(1, actions.unsqueeze(1)).squeeze())
        #actor_loss = -(log_probs * advantage.detach()).mean()
        actor_loss = -(torch.stack(log_probs) * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        
        loss = actor_loss + 0.5 * critic_loss
        
        # Update model
        self.optimizer.zero_grad()
        loss.backward()
        if (total_reward > record):
            for global_param, local_param in zip(global_model.parameters(), self.model.parameters()):
                if global_param.grad is None:
                    global_param.grad = torch.zeros_like(global_param)
                global_param.grad += local_param.grad
        self.optimizer.step()
        
    def update_global_model(self, local_agent):
        # Copy the local agent's gradients to the global model
        for global_param, local_param in zip(self.model.parameters(), local_agent.model.parameters()):
            if local_param.grad is not None:
                if global_param.grad is None:
                    global_param.grad = copy.deepcopy(local_param.grad)
                else:
                    global_param.grad += local_param.grad
        
        # Apply the gradients to update the global model
        self.optimizer.step()
        self.optimizer.zero_grad()
        
    def save_model(self):
        model_dir = os.path.join(os.getcwd(), "model")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, "model.pth")
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
    def load_model(self):
        model_dir = os.path.join(os.getcwd(), "model")
        model_path = os.path.join(model_dir, "model.pth")
        self.model.load_state_dict(torch.load(model_path))
        logging.info(f"Model loaded from {model_path}")
