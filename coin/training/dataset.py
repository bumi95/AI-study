import torch
from torch.utils.data import Dataset, DataLoader
#import numpy as np

class CryptoDataset(Dataset):
    def __init__(self, data, seq_length, target_columns=['close']):
        self.data = data
        self.seq_length = seq_length
        self.target_columns = target_columns
        
        # 시퀀스 데이터 생성
        self.sequences = []
        self.targets = []
        self._prepare_sequences()
        
    def _prepare_sequences(self):
        """데이터를 시퀀스로 변환"""
        for i in range(len(self.data) - self.seq_length):
            sequence = self.data[i:(i + self.seq_length)]
            target = self.data[i + self.seq_length][self.target_columns]
            
            self.sequences.append(sequence)
            self.targets.append(target)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        target = torch.FloatTensor(self.targets[idx])
        return sequence, target

def create_dataloaders(data, seq_length, batch_size=32, train_split=0.8):
    """트레이닝/검증 데이터로더 생성"""
    dataset = CryptoDataset(data, seq_length)
    
    # 데이터 분할
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size]
    )
    
    # 데이터로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader