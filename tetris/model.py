#import torch
import torch.nn as nn
#import torch.nn.functional as F

class A3CModel(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(A3CModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_shape, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=0.3)
        )
        self.policy = nn.Linear(128, num_actions)
        self.value = nn.Linear(128, 1)
        
        # 가중치 초기화
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc(x)
        policy_dist = nn.Softmax(dim=-1)(self.policy(x))
        value = self.value(x)
        return policy_dist, value