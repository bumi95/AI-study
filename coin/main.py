from trading.binance_trader import BinanceTrader
from training.agent import PPOAgent
import numpy as np

def calculate_rsi(prices, period=14):
    """RSI(Relative Strength Index) 계산"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def preprocess_data(df):
    """데이터 전처리"""
    df = df.copy()
    df['close'] = df['close'].astype(float)
    df['returns'] = df['close'].pct_change()
    df['ma7'] = df['close'].rolling(window=7).mean()
    df['ma14'] = df['close'].rolling(window=14).mean()
    df['rsi'] = calculate_rsi(df['close'])
    return df

def main():
    # API 키 설정
    api_key = "xly7tpFQcyVU8dHIVbJpc9ty5Sq9dXPuLgs1Gh6lBMX126SHbAeUF528L8Azg5dI"
    api_secret = "pGCSCtUrJKaYBs0DeiG7tf3ouQfL1NO8FNQ78v1kcBHwhHB95XRU8QqVIyJmoOIW"
    
    # 트레이더 초기화
    trader = BinanceTrader(api_key, api_secret)
    
    # 과거 데이터 가져오기
    data = trader.get_historical_data()
    processed_data = preprocess_data(data)
    
    # 학습 에이전트 초기화
    state_size = 5  # 종가, 거래량, MA7, MA14, RSI
    action_size = 3  # 매수, 매도, 홀딩
    agent = PPOAgent(state_size, action_size)
    
    # 학습 루프
    episodes = 100
    for episode in range(episodes):
        state = processed_data.iloc[30][['close', 'volume', 'ma7', 'ma14', 'rsi']].values.reshape(1, -1)
        action = np.random.randint(0, 3)  # 랜덤 액션 (매수/매도/홀딩)
        
        # 보상 계산 (간단한 예시)
        next_return = processed_data.iloc[31]['returns']
        reward = next_return if action == 1 else -next_return if action == 0 else 0
        
        # 다음 상태 준비
        next_state = processed_data.iloc[31][['close', 'volume', 'ma7', 'ma14', 'rsi']].values.reshape(1, -1)
        
        # 학습
        loss = agent.train_step(state, action, reward, next_state)
        print(f"Episode {episode}, Loss: {loss:.4f}")

if __name__ == "__main__":
    main() 