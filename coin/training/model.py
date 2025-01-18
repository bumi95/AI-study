import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

class RainbowDQN(nn.Module):
    def __init__(self, input_size, hidden_size=128, action_size=3):
        super(RainbowDQN, self).__init__()
        self.num_atoms = 51  # Distributional DQN의 atom 수
        self.v_min = -10
        self.v_max = 10
        self.supports = torch.linspace(self.v_min, self.v_max, self.num_atoms)
        
        # Feature layer
        self.feature = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        
        # Advantage stream
        self.advantage = nn.Sequential(
            NoisyLinear(hidden_size, hidden_size),
            nn.ReLU(),
            NoisyLinear(hidden_size, action_size * self.num_atoms)
        )
        
        # Value stream
        self.value = nn.Sequential(
            NoisyLinear(hidden_size, hidden_size),
            nn.ReLU(),
            NoisyLinear(hidden_size, self.num_atoms)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        features = self.feature(x)
        
        advantage = self.advantage(features).view(batch_size, -1, self.num_atoms)
        value = self.value(features).view(batch_size, 1, self.num_atoms)
        
        # Combine value and advantage using dueling architecture
        q_dist = value + advantage - advantage.mean(dim=1, keepdim=True)
        q_dist = F.softmax(q_dist, dim=-1)
        
        return q_dist
    
    def reset_noise(self):
        for name, module in self.named_modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()