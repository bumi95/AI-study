#import torch
import torch.nn as nn
#import torch.nn.functional as F

class A3CModel(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(A3CModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_shape, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.policy = nn.Linear(256, num_actions)
        self.value = nn.Linear(256, 1)

    def forward(self, x):
        x = self.fc(x)
        policy_dist = nn.Softmax(dim=-1)(self.policy(x))
        value = self.value(x)
        return policy_dist, value
'''
    def __init__(self, input_size, action_size):
        super(A3CModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        
        # Actor network
        self.actor_fc = nn.Linear(128, action_size)
        
        # Critic network
        self.critic_fc = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Actor output (probabilities for actions)
        policy = F.softmax(self.actor_fc(x), dim=-1)
        
        # Critic output (state value)
        value = self.critic_fc(x)
        
        return policy, value
'''
