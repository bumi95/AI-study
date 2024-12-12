import numpy as np
#import pandas as pd
import talib

def calculate_technical_indicators(df):
    """기술적 지표 계산"""
    df = df.copy()
    
    # 이동평균선
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA60'] = df['close'].rolling(window=60).mean()
    
    # RSI
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    
    # MACD
    macd, signal, _ = talib.MACD(df['close'])
    df['MACD'] = macd
    df['MACD_SIGNAL'] = signal
    
    # Bollinger Bands
    df['BB_UPPER'], df['BB_MIDDLE'], df['BB_LOWER'] = talib.BBANDS(
        df['close'], 
        timeperiod=20
    )
    
    # 거래량 지표
    df['VOLUME_MA5'] = df['volume'].rolling(window=5).mean()
    
    return df

def normalize_data(df):
    """데이터 정규화"""
    df = df.copy()
    
    # Min-Max 정규화
    for column in df.columns:
        if column not in ['timestamp', 'date']:
            df[column] = (df[column] - df[column].min()) / \
                        (df[column].max() - df[column].min())
    
    return df

def create_sequences(data, seq_length):
    """시계열 데이터를 시퀀스로 변환"""
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length):
        sequence = data[i:(i + seq_length)]
        target = data[i + seq_length]
        sequences.append(sequence)
        targets.append(target)
        
    return np.array(sequences), np.array(targets)

def calculate_returns(prices):
    """수익률 계산"""
    return np.log(prices).diff()

def calculate_sharpe_ratio(returns, risk_free_rate=0.01):
    """샤프 비율 계산"""
    excess_returns = returns - risk_free_rate/252  # 일간 무위험 수익률
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std() 