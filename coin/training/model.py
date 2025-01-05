import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TradingNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(TradingNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 3)  # 매수, 매도, 홀딩
        
    def forward(self, x):
        # 입력 텐서가 2차원인 경우 sequence_length 차원 추가
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch_size, input_features) -> (batch_size, 1, input_features)
        
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # 마지막 시퀀스 출력만 사용
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

class TradingAgent:
    def __init__(self, state_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TradingNetwork(state_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()
    
    def train_step(self, state, action, reward):
        self.optimizer.zero_grad()
        # numpy array를 float32로 변환하기 전에 배열 타입 확인 및 변환
        state = np.array(state, dtype=np.float32)
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        
        # 입력 차원이 2D가 아닌 경우 차원 추가
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        pred = self.model(state)
        action = torch.LongTensor([action]).to(self.device)
        loss = self.criterion(pred, action)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()