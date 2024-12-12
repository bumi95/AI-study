import torch
import torch.nn as nn
import torch.nn.functional as F

class TradingNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(TradingNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 3)  # 매수, 매도, 홀딩
        
    def forward(self, x):
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
        state = torch.FloatTensor(state).to(self.device)
        
        pred = self.model(state)
        action = torch.LongTensor([action]).to(self.device)
        loss = self.criterion(pred, action)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item() 