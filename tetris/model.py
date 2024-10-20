#import threading
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing
from client_ml import TetrisGameML
import copy
import asyncio
import os

class ActorCriticModel(nn.Module):
    def __init__(self, state_shape, action_space):
        super(ActorCriticModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_shape, 128),
            nn.ReLU(),
            nn.Linear(128, action_space)  # Output action logits and value
        )

    def forward(self, x):
        return self.network(x)

class ActorLearner(multiprocessing.Process):
    def __init__(self, global_model, optimizer, env_id, global_episode_count, max_episode_count, state_shape, action_space):
        super(ActorLearner, self).__init__()
        self.local_model = copy.deepcopy(global_model)
        self.global_model = global_model
        self.optimizer = optimizer
        self.env_id = env_id
        self.global_episode_count = global_episode_count
        self.max_episode_count = max_episode_count
        self.state_shape = state_shape
        self.action_space = action_space
        self.record = 0

    def run(self):
        self.env = TetrisGameML()
        while self.global_episode_count.value < self.max_episode_count:
            self.train_episode()

    def train_episode(self):
        #print(f"Environment {self.env_id}: Starting episode {self.global_episode_count.value}")
        state = torch.tensor(self.env.reset(), dtype=torch.float32)
        done = False
        total_score = 0
        states, actions, rewards, values = [], [], [], []

        while not done:
            logits_values = self.local_model(state)
            policy_logits = logits_values[:-1]
            value = logits_values[-1]
            policy_probs = torch.softmax(policy_logits, dim=0).detach().numpy()
            action = np.random.choice(len(policy_probs), p=policy_probs)
            #print(f"policy_probs len : {len(policy_probs)}, action : {action}")
            step_result = asyncio.run(self.env.step(self.env.action_space[action]))
            #step_result = self.env.step(self.env.action_space[action])  # Ensure proper unpacking of step result
            #if asyncio.iscoroutine(step_result):
            #    step_result = asyncio.run(step_result)
                
            new_state, reward, done, info = step_result  # Ensure proper unpacking of step result
            print(f"reward : {reward} done : {done}, info : {info}")
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value)

            if not isinstance(new_state, torch.Tensor):
                state = torch.tensor(new_state, dtype=torch.float32)
            else:
                state = new_state
            total_score += info["score"]

        # Compute returns and advantages, then update global model
        self.update_global_model(states, actions, rewards, values)
        with self.global_episode_count.get_lock():
            self.global_episode_count.value += 1
        
        if total_score > self.record:
            self.record = total_score
            print(f"Environment {self.env_id}, 총 수행 횟수 {self.global_episode_count.value} New record: {self.record}")
            self.save_model()
        else:
            print(f"Environment {self.env_id}, 총 수행 횟수 {self.global_episode_count.value}")

    def update_global_model(self, states, actions, rewards, values):
        # Calculate discounted rewards and advantages
        discounted_rewards = self.compute_discounted_rewards(rewards)
        advantages = discounted_rewards - torch.stack(values)
        total_loss = 0
        self.optimizer.zero_grad()

        for i in range(len(states)):
            state = states[i]
            logits_values = self.local_model(state)
            policy_logits = logits_values[:-1]
            #value = logits_values[-1]
            advantage = advantages[i]

            policy_loss = self.compute_policy_loss(policy_logits, actions[i], advantage)
            value_loss = (advantage ** 2) * 0.5  # Scaled value loss
            total_loss += (policy_loss + value_loss)

        if torch.isnan(total_loss):
            print(f"NaN detected in total_loss for Environment {self.env_id}, skipping gradient update.")
            return
        
        total_loss.backward()
        for global_param, local_param in zip(self.global_model.parameters(), self.local_model.parameters()):
            if global_param.grad is None:
                global_param.grad = torch.zeros_like(global_param)
            global_param.grad.add_(local_param.grad)
        self.optimizer.step()
        self.local_model.load_state_dict(self.global_model.state_dict())

    def compute_discounted_rewards(self, rewards, gamma=0.99):
        discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * gamma + rewards[t]
            discounted_rewards[t] = running_add
        return torch.tensor(discounted_rewards, dtype=torch.float32)

    def compute_policy_loss(self, policy_logits, action, advantage):
        policy = torch.softmax(policy_logits, dim=0)
        action_prob = policy[action]
        return -torch.log(action_prob) * advantage
    
    def save_model(self):
        model_dir = os.path.join(os.getcwd(), 'model')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, 'model.pth')
        torch.save(self.global_model.state_dict(), model_path)
        #print(f"Model saved to {model_path}")

def main():
    num_actors = 5
    max_episode_count = 1000
    temp_env = TetrisGameML()  # Temporary environment instance to access state and action space
    global_model = ActorCriticModel(temp_env.get_state().shape[0], len(temp_env.action_space) + 1)
    global_model.share_memory()
    optimizer = optim.Adam(global_model.parameters(), lr=0.001)
    global_episode_count = multiprocessing.Value('i', 0, lock=True)

    # Create and start actor-learner threads
    actors = [ActorLearner(global_model, optimizer, i, global_episode_count, max_episode_count, temp_env.get_state().shape[0], len(temp_env.action_space)) for i in range(num_actors)]
    for actor in actors:
        actor.start()
    for actor in actors:
        actor.join()

if __name__ == "__main__":
    main()