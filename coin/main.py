from trading.binance_trader import BinanceTrader
from training.agent import RainbowAgent
import numpy as np
import pandas as pd
from collections import deque
import time
import torch
from decimal import Decimal, ROUND_DOWN
#import sys

is_running = True

def calculate_rsi(prices, period=14):
    """RSI(Relative Strength Index) 계산"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.bfill()  # fillna 대신 bfill 사용

class TradingEnvironment:
    def __init__(self, window_size=60):
        self.window_size = window_size
        self.price_history = deque(maxlen=window_size)
        self.volume_history = deque(maxlen=window_size)
        self.volume_ratio_history = deque(maxlen=window_size)
        self.features = deque(maxlen=window_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def update(self, price_data, position_info):
        """새로운 가격 데이터로 환경 업데이트"""
        # Kline 데이터 추출
        kline = price_data['k']
        current_price = float(kline['c'])    # 현재가 (종가)
        open_price = float(kline['o'])       # 시가
        high_price = float(kline['h'])       # 고가
        low_price = float(kline['l'])        # 저가
        volume = float(kline['v'])           # 거래량
        quote_volume = float(kline['q'])     # 거래대금
        trades_count = int(kline['n'])       # 거래 횟수
        taker_volume = float(kline['V'])     # Taker 거래량
        taker_quote_volume = float(kline['Q'])  # Taker 거래대금
        
        # 핵심 지표 계산
        price_range = (high_price - low_price) / low_price  # 변동폭
        body_size = abs(current_price - open_price) / open_price  # 캔들 몸통 크기
        price_direction = 1 if current_price >= open_price else -1
        
        # 거래량 관련 지표
        volume_ratio = volume * price_direction  # 방향성이 반영된 거래량
        taker_ratio = taker_volume / volume if volume > 0 else 0  # Taker 비율
        avg_trade_size = volume / trades_count if trades_count > 0 else 0  # 평균 거래 크기
        quote_avg_price = quote_volume / volume if volume > 0 else current_price  # 평균 거래가격
        taker_dominance = taker_quote_volume / quote_volume if quote_volume > 0 else 0  # Taker의 영향력
        
        self.price_history.append(current_price)
        self.volume_history.append(volume)
        self.volume_ratio_history.append(volume_ratio)
        
        if len(self.price_history) >= 14:
            prices = pd.Series(list(self.price_history))
            volumes = pd.Series(list(self.volume_history))
            volume_ratios = pd.Series(list(self.volume_ratio_history))
            
            # 5분 이동평균만 사용
            ma5 = prices.rolling(window=5).mean().bfill()   # 5분 이동평균
            rsi = calculate_rsi(prices, period=14)          # RSI는 14포인트로 유지
            
            # 거래량 이동평균
            vol_ma5 = volumes.rolling(window=5).mean().bfill()
            vol_ratio_ma5 = volume_ratios.rolling(window=5).mean().bfill()
            
            # 상태 정규화
            epsilon = 1e-8
            price_std = prices.std() + epsilon
            vol_std = volumes.std() + epsilon
            
            # 핵심 특성 정규화
            normalized_price = (current_price - prices.mean()) / price_std
            normalized_ma5 = (ma5.iloc[-1] - ma5.mean()) / (ma5.std() + epsilon)
            normalized_rsi = rsi.iloc[-1] / 100.0
            normalized_range = price_range / 0.001  # 0.1% 기준
            normalized_volume = (volume - volumes.mean()) / vol_std
            
            # 거래 활동 지표 정규화
            normalized_taker = taker_ratio - 0.5  # 0.5를 중심으로 정규화
            normalized_trade_size = (avg_trade_size / (vol_ma5.iloc[-1] / trades_count) - 1) if trades_count > 0 else 0
            normalized_price_impact = ((current_price / quote_avg_price) - 1) / 0.001  # 0.1% 기준
            normalized_vol_ratio = (volume_ratios.iloc[-1] - vol_ratio_ma5.iloc[-1]) / (vol_ratio_ma5.std() + epsilon)
            
            # 포지션 정보
            position_size = float(position_info['size']) if position_info else 0.0
            
            self.features.append([
                normalized_price,        # 1. 정규화된 현재가
                normalized_ma5,         # 2. 정규화된 5분 이동평균
                normalized_rsi,         # 3. 정규화된 RSI
                normalized_range,       # 4. 정규화된 가격 변동폭
                normalized_volume,      # 5. 정규화된 거래량
                normalized_taker,       # 6. 정규화된 Taker 비율
                body_size / 0.001,      # 7. 정규화된 캔들 몸통 크기
                normalized_trade_size,  # 8. 정규화된 평균 거래 크기
                normalized_price_impact,# 9. 가격 영향도
                taker_dominance,       # 10. Taker 지배도
                normalized_vol_ratio,   # 11. 방향성 반영된 거래량 모멘텀
                float(position_info['pnl']),  # 12. 현재 미실현 손익
                position_size,          # 13. 현재 포지션 크기
                float(position_info['balance'] / position_info['initial_balance']),  # 14. 자산 비율
            ])
    
    def get_state(self):
        """현재 상태 반환"""
        if len(self.features) < 1:
            return None
        state = np.array(list(self.features)[-1], dtype=np.float32)
        if np.any(np.isnan(state)):
            return None
        return state

class TradingBot:
    def __init__(self, api_key, api_secret, symbol="BTCUSDT", leverage=3, initial_balance=1000.0):
        self.env = TradingEnvironment(window_size=60)
        self.trader = BinanceTrader(api_key, api_secret, symbol=symbol)
        self.trader.set_leverage(leverage)
        self.leverage = leverage
        
        # Rainbow DQN 에이전트 초기화
        state_size = 14  # 상태 크기 증가
        action_size = 3  # LONG(0), SHORT(1), CLEAR_ALL(2)
        self.agent = RainbowAgent(
            state_size=state_size,
            action_size=action_size,
            lr=1e-5,  # 더 작은 학습률로 조정
            gamma=0.95,  # 감가율 조정
            batch_size=64,  # 배치 크기 증가
            n_step=1  # n-step 감소
        )
        
        # 모델 로드
        self.agent.load_model()
        
        # Decimal로 초기화
        self.initial_balance = Decimal(str(initial_balance))
        self.virtual_balance = Decimal(str(initial_balance))
        self.virtual_position = Decimal('0')
        self.virtual_entry_price = Decimal('0')
        self.virtual_pnl = Decimal('0')
        
        # 수수료 설정
        self.maker_fee = Decimal('0.0002')
        self.taker_fee = Decimal('0.0004')
        
        self.last_state = None
        self.last_action = None
        self.last_position = None
        
    def calculate_fee(self, position_size, price, leverage, is_taker=True):
        """거래 수수료 계산"""
        fee_rate = self.taker_fee if is_taker else self.maker_fee
        position_size = Decimal(str(abs(position_size)))
        price = Decimal(str(price))
        return (position_size * price * fee_rate * leverage).quantize(Decimal('0.00000001'), rounding=ROUND_DOWN)
    
    def calculate_virtual_pnl(self, current_price):
        """가상 포지션의 미실현 손익 계산 (수수료 포함)"""
        if self.virtual_position == 0:
            return Decimal('0')
        
        try:
            current_price = Decimal(str(current_price))
            position_size = Decimal(str(abs(self.virtual_position)))
            
            # 레버리지를 고려한 실제 포지션 가치 계산
            position_value = position_size * self.virtual_entry_price * self.leverage
            current_value = position_size * current_price * self.leverage
            
            # 수수료 계산 (레버리지 포함된 금액 기준)
            entry_fee = self.calculate_fee(self.virtual_position, self.virtual_entry_price, self.leverage)
            exit_fee = self.calculate_fee(self.virtual_position, current_price, self.leverage)
            total_fee = entry_fee + exit_fee
            
            if self.virtual_position > 0:  # 롱 포지션
                pnl = current_value - position_value - total_fee
            else:  # 숏 포지션
                pnl = position_value - current_value - total_fee
            
            # 디버그 로깅
            #print(f"\nPnL 계산 상세:")
            #print(f"Position Size: {position_size}")
            #print(f"Entry Price: {self.virtual_entry_price}")
            #print(f"Current Price: {current_price}")
            #print(f"Position Value: {position_value}")
            #print(f"Current Value: {current_value}")
            #print(f"Fees: {total_fee}")
            #print(f"Final PnL: {pnl}\n")
            
            return pnl.quantize(Decimal('0.00000001'), rounding=ROUND_DOWN)
        
        except Exception as e:
            print(f"PnL 계산 오류: {e}")
            return Decimal('0')
    
    def execute_virtual_trade(self, action, current_price):
        """가상 거래 실행"""
        # float을 Decimal로 변환
        current_price = Decimal(str(current_price))
        
        # 현재 총 자산 계산 (현재 잔고 + 미실현 손익)
        total_equity = self.virtual_balance + self.calculate_virtual_pnl(current_price)
        
        # 파산 체크 (총 자산이 초기 자산의 1% 미만이면 파산으로 간주)
        if total_equity < (self.initial_balance * Decimal('0.01')):
            print(f"\n=== 거래 종료 ===")
            print(f"초기 자산: {float(self.initial_balance):.2f} USDT")
            print(f"최종 자산: {float(total_equity):.2f} USDT")
            print(f"수익률: {((total_equity/self.initial_balance) - 1) * 100:.2f}%")
            print(f"파산으로 인한 프로그램 종료")
            global is_running
            is_running = False
            
        last_virtual_position = {
            'amount': float(self.virtual_position),
            'entry_price': float(self.virtual_entry_price),
            'unrealized_pnl': float(self.virtual_pnl),
            'balance': float(self.virtual_balance),
            'total_value': float(self.virtual_balance + self.virtual_pnl),
            'leverage': float(self.leverage)
        }
        
        # 레버리지를 고려한 최대 거래 가능 수량 계산
        max_quantity = (total_equity * self.leverage / current_price).quantize(Decimal('0.00001'), rounding=ROUND_DOWN)
        
        if action == 0:  # LONG
            if self.virtual_position < 0:  # 숏 포지션 청산
                self.virtual_balance += self.calculate_virtual_pnl(current_price)
                self.virtual_position = Decimal('0')
            if self.virtual_position <= 0:  # 롱 포지션 진입
                self.virtual_position = max_quantity
                self.virtual_entry_price = current_price
                entry_fee = self.calculate_fee(self.virtual_position, current_price, self.leverage)
                self.virtual_balance -= entry_fee
        
        elif action == 1:  # SHORT
            if self.virtual_position > 0:  # 롱 포지션 청산
                self.virtual_balance += self.calculate_virtual_pnl(current_price)
                self.virtual_position = Decimal('0')
            if self.virtual_position >= 0:  # 숏 포지션 진입
                self.virtual_position = -max_quantity
                self.virtual_entry_price = current_price
                entry_fee = self.calculate_fee(self.virtual_position, current_price, self.leverage)
                self.virtual_balance -= entry_fee
        
        elif action == 2:  # CLEAR_ALL
            if self.virtual_position != 0:
                self.virtual_balance += self.calculate_virtual_pnl(current_price)
                self.virtual_position = Decimal('0')
                self.virtual_entry_price = Decimal('0')
        
        # 새로운 미실현 손익 계산
        self.virtual_pnl = self.calculate_virtual_pnl(current_price)
        
        return last_virtual_position
        
    def calculate_reward(self, action, virtual_position):
        """보상 계산"""
        # CLEAR_ALL 액션의 경우
        if action == 2 and self.last_position:
            # 이전 포지션과 현재 포지션의 차이로 실제 손익 계산
            prev_balance = Decimal(str(self.last_position['balance']))
            current_balance = Decimal(str(virtual_position['balance']))
            actual_profit = current_balance - prev_balance
            
            # 이전 포지션 크기 기준으로 ROI 계산
            position_size = abs(float(self.last_position['amount']) * float(self.last_position['entry_price']))
            if position_size > 0:
                roi = float(actual_profit / Decimal(str(position_size)))
                return roi
        
        # LONG 또는 SHORT 포지션의 경우
        reward = Decimal(str(virtual_position['unrealized_pnl']))
        if reward != 0:
            position_size = abs(self.virtual_position * self.virtual_entry_price)
            if position_size > 0:
                roi = float(reward / Decimal(str(position_size)))
                return roi
        
        # 변동이 없을 때의 페널티
        asset_ratio = (self.virtual_balance + self.virtual_pnl) / self.initial_balance
        base_penalty = Decimal('0.001')
        return float(-base_penalty * (Decimal('1') / asset_ratio))
    
    def handle_ticker(self, msg):
        """실시간 데이터 처리"""
        current_price = self.trader.get_current_price()  # 현재 가격을 먼저 저장
        
        position_info = {
            'size': self.virtual_position,
            'pnl': self.virtual_pnl,
            'balance': self.virtual_balance,
            'initial_balance': self.initial_balance
        }
        
        self.env.update(msg, position_info)
        current_state = self.env.get_state()
        
        if current_state is not None:
            # 이전 행동의 보상 계산 (있는 경우)
            if self.last_state is not None and self.last_action is not None:
                # 현재 가격으로 이전 포지션의 PnL 계산
                virtual_position = self.calculate_position_status(current_price)
                reward = self.calculate_reward(self.last_action, virtual_position)
                
                # 경험 저장
                self.agent.remember(self.last_state, self.last_action, reward, current_state, False)
                
                # 로깅
                if self.last_position:
                    print(f"Price: {current_price}, Last Action: {['LONG', 'SHORT', 'CLEAR'][self.last_action]}, "
                          f"Position: {virtual_position['amount']:.4f}, "
                          f"Entry: {virtual_position['entry_price']:.2f}, "
                          f"PnL: {virtual_position['unrealized_pnl']:.4f}, "
                          f"Balance: {virtual_position['balance']:.4f} USDT, "
                          f"Total: {virtual_position['total_value']:.4f} USDT, "
                          f"ROI: {reward:.4%}")
            
            # 새로운 행동 선택 및 실행
            state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(self.env.device)
            with torch.no_grad():
                q_dist = self.agent.policy_net(state_tensor)
                action = (q_dist * self.agent.supports.expand_as(q_dist)).sum(2).max(1)[1].item()
            
            # 현재 가격으로 가상 거래 실행
            virtual_position = self.execute_virtual_trade(action, current_price)
            
            # 현재 상태, 행동, 포지션 저장
            self.last_state = current_state
            self.last_action = action
            self.last_position = virtual_position
            
            # 학습 수행
            if self.agent.memory.size_now >= self.agent.batch_size:
                loss = self.agent.train_step()
                if loss is not None:
                    print(f"Loss: {loss:.4f}")
    
    def calculate_position_status(self, current_price):
        """현재 포지션 상태 계산"""
        return {
            'amount': float(self.virtual_position),
            'entry_price': float(self.virtual_entry_price),
            'unrealized_pnl': float(self.calculate_virtual_pnl(current_price)),
            'balance': float(self.virtual_balance),
            'total_value': float(self.virtual_balance + self.calculate_virtual_pnl(current_price)),
            'leverage': float(self.leverage)
        }
    
    def start(self):
        """거래 봇 시작"""
        print("실시간 데이터 수신 시작...")
        print(f"현재 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"거래 심볼: {self.trader.symbol}")
        print(f"레버리지: {self.leverage}x")
        print(f"초기 잔고: {self.initial_balance} USDT")
        
        self.trader.start_ticker_socket(self.handle_ticker)
        
        # 웹소켓 연결 확인
        time.sleep(5)  # 연결 대기
        if self.trader.current_price is None:
            print("경고: 실시간 데이터가 수신되지 않고 있습니다.")
        else:
            print(f"최우선 매수 호가 수신 중: {self.trader.current_price}")
    
    def stop(self):
        """거래 봇 종료"""
        print("프로그램 종료 중...")
        self.agent.save_model()
        self.trader.stop_ticker_socket()
        print("프로그램이 안전하게 종료되었습니다.")

def main():
    # API 키 설정
    api_key = "***"
    api_secret = "***"
    
    # 거래 봇 초기화 및 실행
    bot = TradingBot(
        api_key=api_key,
        api_secret=api_secret,
        leverage=3,
        initial_balance=15000.0
    )
    
    #try:
    bot.start()
    while True:
        if not is_running:
            break
        time.sleep(1)
    #except KeyboardInterrupt:
    bot.stop()
    print("프로그램이 종료되었습니다.")

if __name__ == "__main__":
    main() 