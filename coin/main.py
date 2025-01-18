from trading.binance_trader import BinanceTrader
from training.agent import RainbowAgent
import numpy as np
import pandas as pd
from collections import deque
import time
import torch

def calculate_rsi(prices, period=14):
    """RSI(Relative Strength Index) 계산"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

class TradingEnvironment:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.price_history = deque(maxlen=window_size)
        self.volume_history = deque(maxlen=window_size)
        self.features = deque(maxlen=window_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def update(self, price_data):
        """새로운 가격 데이터로 환경 업데이트"""
        self.price_history.append(float(price_data['b']))  # 최우선 매수호가 사용
        
        if len(self.price_history) >= 14:  # RSI 계산에 필요한 최소 데이터
            prices = pd.Series(list(self.price_history))
            ma7 = prices.rolling(window=7).mean().fillna(method='bfill')
            ma14 = prices.rolling(window=14).mean().fillna(method='bfill')
            rsi = calculate_rsi(prices).fillna(method='bfill')
            
            current_price = prices.iloc[-1]
            
            # 상태 정규화
            normalized_price = (current_price - prices.mean()) / (prices.std() + 1e-8)
            normalized_ma7 = (ma7.iloc[-1] - ma7.mean()) / (ma7.std() + 1e-8)
            normalized_ma14 = (ma14.iloc[-1] - ma14.mean()) / (ma14.std() + 1e-8)
            normalized_rsi = rsi.iloc[-1] / 100.0  # RSI는 0-100 범위
            
            self.features.append([
                normalized_price,    # 정규화된 현재가
                normalized_ma7,      # 정규화된 MA7
                normalized_ma14,     # 정규화된 MA14
                normalized_rsi,      # 정규화된 RSI
                1.0                  # 바이어스 항
            ])
    
    def get_state(self):
        """현재 상태 반환"""
        if len(self.features) < 1:
            return None
        return np.array(list(self.features)[-1], dtype=np.float32)

class TradingBot:
    def __init__(self, api_key, api_secret, symbol="BTCUSDT", leverage=3, initial_balance=1000.0):
        self.env = TradingEnvironment(window_size=30)
        self.trader = BinanceTrader(api_key, api_secret, symbol=symbol)
        self.trader.set_leverage(leverage)
        
        # Rainbow DQN 에이전트 초기화
        state_size = 5  # 정규화된 현재가, MA7, MA14, RSI, 바이어스
        action_size = 3  # LONG(0), SHORT(1), CLEAR_ALL(2)
        self.agent = RainbowAgent(
            state_size=state_size,
            action_size=action_size,
            lr=1e-4,
            gamma=0.99,
            batch_size=32,
            n_step=3
        )
        
        self.quantity = 0.001  # 거래 수량
        self.leverage = leverage
        
        # 가상 계좌 정보
        self.initial_balance = initial_balance  # 초기 USDT 잔고
        self.virtual_balance = initial_balance  # 현재 USDT 잔고
        self.virtual_position = 0.0  # 현재 포지션 크기
        self.virtual_entry_price = 0.0  # 진입 가격
        self.virtual_pnl = 0.0  # 미실현 손익
        
        # 수수료 설정 (바이낸스 선물 기준)
        self.maker_fee = 0.0002  # 0.02%
        self.taker_fee = 0.0005  # 0.05%
        
    def calculate_fee(self, position_size, price, is_taker=True):
        """거래 수수료 계산"""
        fee_rate = self.taker_fee if is_taker else self.maker_fee
        return abs(position_size) * price * fee_rate
    
    def calculate_virtual_pnl(self, current_price):
        """가상 포지션의 미실현 손익 계산 (수수료 포함)"""
        if self.virtual_position == 0:
            return 0.0
        
        position_value = abs(self.virtual_position) * self.virtual_entry_price
        current_value = abs(self.virtual_position) * current_price
        
        # 진입/청산 시 발생하는 수수료 (레버리지와 무관)
        entry_fee = self.calculate_fee(self.virtual_position, self.virtual_entry_price)
        exit_fee = self.calculate_fee(self.virtual_position, current_price)
        total_fee = entry_fee + exit_fee
        
        if self.virtual_position > 0:  # 롱 포지션
            # 레버리지는 손익에만 적용, 수수료는 레버리지 적용 안 함
            pnl = (current_value - position_value) * self.leverage
        else:  # 숏 포지션
            pnl = (position_value - current_value) * self.leverage
            
        # 최종 손익에서 수수료 차감
        return pnl - total_fee
    
    def execute_virtual_trade(self, action, current_price):
        """가상 거래 실행"""
        old_position = self.virtual_position
        old_pnl = self.calculate_virtual_pnl(current_price)
        
        if action == 0:  # LONG 포지션 진입
            if self.virtual_position < 0:  # 숏 포지션 청산
                exit_fee = self.calculate_fee(self.virtual_position, current_price)
                self.virtual_balance += old_pnl - exit_fee
                self.virtual_position = 0
            if self.virtual_position <= 0:  # 롱 포지션 진입
                self.virtual_position = self.quantity
                self.virtual_entry_price = current_price
                entry_fee = self.calculate_fee(self.virtual_position, current_price)
                self.virtual_balance -= entry_fee
                
        elif action == 1:  # SHORT 포지션 진입
            if self.virtual_position > 0:  # 롱 포지션 청산
                exit_fee = self.calculate_fee(self.virtual_position, current_price)
                self.virtual_balance += old_pnl - exit_fee
                self.virtual_position = 0
            if self.virtual_position >= 0:  # 숏 포지션 진입
                self.virtual_position = -self.quantity
                self.virtual_entry_price = current_price
                entry_fee = self.calculate_fee(self.virtual_position, current_price)
                self.virtual_balance -= entry_fee
                
        elif action == 2:  # CLEAR_ALL
            if self.virtual_position != 0:
                exit_fee = self.calculate_fee(self.virtual_position, current_price)
                self.virtual_balance += old_pnl - exit_fee
                self.virtual_position = 0
                self.virtual_entry_price = 0
        
        # 새로운 미실현 손익 계산
        self.virtual_pnl = self.calculate_virtual_pnl(current_price)
        
        return {
            'amount': self.virtual_position,
            'entry_price': self.virtual_entry_price,
            'unrealized_pnl': self.virtual_pnl,
            'balance': self.virtual_balance,
            'total_value': self.virtual_balance + self.virtual_pnl,
            'leverage': self.leverage
        }
        
    def handle_ticker(self, msg):
        """실시간 데이터 처리"""
        self.env.update(msg)
        state = self.env.get_state()
        
        if state is not None:
            current_price = float(msg['b'])
            
            # 상태를 torch tensor로 변환
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.env.device)
            
            # 에이전트로부터 행동 선택
            with torch.no_grad():
                q_dist = self.agent.policy_net(state_tensor)
                action = (q_dist * self.agent.supports.expand_as(q_dist)).sum(2).max(1)[1].item()
            
            # 가상 거래 실행
            virtual_position = self.execute_virtual_trade(action, current_price)
            
            # 보상 계산 (수수료를 고려한 순수익)
            reward = virtual_position['unrealized_pnl']
            if reward != 0:
                # ROI 기반 보상 계산
                roi = reward / (self.initial_balance * self.leverage)
                
                # 청산 액션에 대한 추가 보상 설정
                if action == 2 and abs(virtual_position['amount']) > 0:
                    if roi > 0:
                        roi *= 1.2  # 20% 추가 보상
                    else:
                        roi *= 0.9  # 10% 페널티
                reward = roi
            
            # 경험 저장
            self.agent.remember(state, action, reward, state, False)
            
            # 학습 수행
            if self.agent.memory.size_now >= self.agent.batch_size:
                loss = self.agent.train_step()
                
                # 로깅
                if loss is not None:
                    print(f"Price: {current_price}, Action: {['LONG', 'SHORT', 'CLEAR'][action]}, "
                          f"Position: {virtual_position['amount']}, "
                          f"PnL: {virtual_position['unrealized_pnl']:.4f}, "
                          f"Balance: {virtual_position['balance']:.4f} USDT, "
                          f"Total: {virtual_position['total_value']:.4f} USDT, "
                          f"ROI: {reward:.4%}, Loss: {loss:.4f}")
                    
                # 주기적으로 타겟 네트워크 업데이트
                if self.agent.memory.size_now % 100 == 0:
                    self.agent.update_target_network()
    
    def start(self):
        """거래 봇 시작"""
        print("실시간 데이터 수신 시작...")
        self.trader.start_ticker_socket(self.handle_ticker)
    
    def stop(self):
        """거래 봇 종료"""
        print("프로그램 종료 중...")
        self.trader.stop_ticker_socket()
        print("프로그램이 안전하게 종료되었습니다.")

def main():
    # API 키 설정
    api_key = "your_api_key"
    api_secret = "your_api_secret"
    
    # 거래 봇 초기화 및 실행
    bot = TradingBot(api_key, api_secret, leverage=3)
    
    try:
        bot.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        bot.stop()

if __name__ == "__main__":
    main() 