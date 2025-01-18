from binance.client import Client
from binance.enums import *
from binance import ThreadedWebsocketManager
import pandas as pd
import time

class BinanceTrader:
    def __init__(self, api_key, api_secret, symbol="BTCUSDT"):
        self.client = Client(api_key, api_secret)
        self.symbol = symbol.lower()
        self.twm = ThreadedWebsocketManager(api_key=api_key, api_secret=api_secret)
        self.current_price = None
        self.last_update_time = None
        self.twm.start()
        
    def start_ticker_socket(self, callback):
        """실시간 선물 틱데이터 수신 시작"""
        def handle_message(msg):
            if msg.get('e') == 'bookTicker':
                self.current_price = float(msg['b'])
                self.last_update_time = time.time()
                if callback:
                    callback(msg)
        
        # 선물 마켓 스트림 사용
        self.twm.start_symbol_ticker_socket(
            callback=handle_message,
            symbol=self.symbol
        )
        
    def stop_ticker_socket(self):
        """웹 소켓 종료"""
        self.twm.stop()
    
    def place_order(self, side, quantity, order_type=ORDER_TYPE_MARKET):
        """선물 주문 실행"""
        try:
            order = self.client.futures_create_order(
                symbol=self.symbol.upper(),
                side=side,
                type=order_type,
                quantity=quantity
            )
            return order
        except Exception as e:
            print(f"선물 주문 실패: {e}")
            return None
            
    def get_current_price(self):
        """선물 현재가 조회"""
        if self.current_price is None or time.time() - self.last_update_time > 5:
            ticker = self.client.futures_symbol_ticker(symbol=self.symbol.upper())
            return float(ticker['price'])
        return self.current_price
    
    def get_position(self):
        """선물 포지션 조회"""
        try:
            position = self.client.futures_position_information(symbol=self.symbol.upper())[0]
            return {
                'amount': float(position['positionAmt']),
                'entry_price': float(position['entryPrice']),
                'unrealized_pnl': float(position['unRealizedProfit']),
                'leverage': float(position['leverage']),
                'liquidation_price': float(position['liquidationPrice'])
            }
        except Exception as e:
            print(f"선물 포지션 조회 실패: {e}")
            return None
    
    def set_leverage(self, leverage):
        """선물 레버리지 설정"""
        try:
            self.client.futures_change_leverage(
                symbol=self.symbol.upper(),
                leverage=leverage
            )
            # 격리 마진 모드 설정
            self.client.futures_change_margin_type(
                symbol=self.symbol.upper(),
                marginType='ISOLATED'
            )
            return True
        except Exception as e:
            print(f"레버리지 설정 실패: {e}")
            return False