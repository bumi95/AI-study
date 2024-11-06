#import torch
import torch.nn as nn
#import torch.nn.functional as F

class A3CModel(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(A3CModel, self).__init__()
        
        # 특징 추출기 강화
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_shape, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.01)
        )
        
        # 정책 네트워크
        self.policy_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, num_actions)
        )
        
        # 가치 네트워크
        self.value_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.feature_extractor(x)
        policy_logits = self.policy_net(features)
        policy_dist = nn.Softmax(dim=-1)(policy_logits)
        value = self.value_net(features)
        return policy_dist, value
