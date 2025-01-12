import torch
import torch.nn as nn
import torch.optim as optim
from training.model import ActorCritic
import numpy as np

class PPOAgent:
    def __init__(self, state_size, action_size, lr=3e-4, gamma=0.99, eps_clip=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(state_size, action_size=action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.criterion = nn.MSELoss()

    def compute_advantages(self, rewards, values, next_value):
        advantages = []
        advantage = 0
        for r, v in zip(reversed(rewards), reversed(values)):
            td_error = r + self.gamma * next_value - v
            advantage = td_error + self.gamma * advantage
            advantages.insert(0, advantage)
            next_value = v
        return advantages

    def train_step(self, states, action, reward, next_states):
        # 입력 데이터가 단일 샘플인 경우를 처리
        if not isinstance(states, list):
            states = [states]
            next_states = [next_states]
        
        # numpy array로 변환 및 차원 확인
        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        
        # 차원 추가가 필요한 경우
        if len(states.shape) == 1:
            states = states.reshape(1, -1)
        if len(next_states.shape) == 1:
            next_states = next_states.reshape(1, -1)
        
        # 텐서 변환
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        action = torch.tensor([action], dtype=torch.int64).to(self.device)  # 단일 액션을 리스트로 감싸기
        reward = torch.tensor([reward], dtype=torch.float32).to(self.device)  # 단일 보상을 리스트로 감싸기
        
        # Compute current values and action probabilities
        action_probs, state_values = self.model(states)
        action_log_probs = torch.log(action_probs.gather(1, action.unsqueeze(1)))

        # Compute advantages
        _, next_value = self.model(next_states[0])
        advantages = self.compute_advantages(reward, state_values, next_value.item())

        # Compute loss
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        ratio = torch.exp(action_log_probs - action_log_probs.detach())
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = self.criterion(state_values, reward + self.gamma * next_value)
        
        loss = actor_loss + 0.5 * critic_loss

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item() 