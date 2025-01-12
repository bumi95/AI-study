import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, input_size, hidden_size=64, action_size=3):
        super(ActorCritic, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.actor_fc = nn.Linear(hidden_size, action_size)
        self.critic_fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        
        action_probs = F.softmax(self.actor_fc(x), dim=1)
        state_value = self.critic_fc(x)
        
        return action_probs, state_value