import torch
import numpy as np
import asyncio
from client_ml import TetrisGameML

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_space):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.actor = nn.Linear(128, len(action_space))
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy_logits = self.actor(x)
        value = self.critic(x)
        return policy_logits, value

class A3CAgent:
    def __init__(self, input_dim, action_space, lr=1e-4):
        self.model = ActorCritic(input_dim, action_space)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.action_space = action_space

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        policy_logits, _ = self.model(state)
        policy = F.softmax(policy_logits, dim=1)
        action = np.random.choice(len(self.action_space), p=policy.detach().numpy().flatten())
        return self.action_space[action]

    def learn(self, state, action, reward, next_state, done):
        """
        에이전트가 주어진 상태에서 학습을 수행하는 함수입니다.
        매개변수:
        state (array-like): 현재 상태를 나타내는 입력 데이터.
        action (int): 에이전트가 취한 행동.
        reward (float): 행동에 대한 보상.
        next_state (array-like): 다음 상태를 나타내는 입력 데이터.
        done (bool): 에피소드 종료 여부를 나타내는 플래그.
        설명:
        이 함수는 주어진 상태, 행동, 보상, 다음 상태 및 에피소드 종료 여부를 사용하여
        에이전트의 정책 및 가치 함수를 업데이트합니다. 손실 함수는 정책 손실과 가치 손실의
        합으로 정의되며, 역전파를 통해 모델의 가중치를 업데이트합니다.
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        action = torch.tensor(action).unsqueeze(0)
        reward = torch.tensor(reward).unsqueeze(0)
        done = torch.tensor(done).unsqueeze(0)

        _, value = self.model(state)
        _, next_value = self.model(next_state)

        target = reward + (~done) * 0.99 * next_value
        advantage = target - value

        policy_logits, value = self.model(state)
        policy = F.softmax(policy_logits, dim=1)
        log_policy = torch.log(policy)
        selected_log_policy = log_policy.gather(1, action.unsqueeze(1))

        actor_loss = -(selected_log_policy * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

async def train_agent():
    game = TetrisGameML()
    agent = A3CAgent(input_dim=game.get_state().shape[0], action_space=game.action_space)

    for episode in range(1000):
        state = game.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = await game.step(action)
            agent.learn(state, game.action_space.index(action), reward, next_state, done)
            state = next_state
            total_reward += reward

        print(f"Episode {episode} - Total Reward: {total_reward}")

if __name__ == "__main__":
    asyncio.run(train_agent())