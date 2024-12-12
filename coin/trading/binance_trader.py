from binance.client import Client
from binance.enums import *
import pandas as pd
#import numpy as np

class BinanceTrader:
    def __init__(self, api_key, api_secret, symbol="BTCUSDT"):
        self.client = Client(api_key, api_secret)
        self.symbol = symbol
        
    def get_historical_data(self, interval="1h", limit=1000):
        """과거 가격 데이터 가져오기"""
        klines = self.client.get_klines(
            symbol=self.symbol,
            interval=interval,
            limit=limit
        )
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close',
            'volume', 'close_time', 'quote_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    
    def place_order(self, side, quantity, order_type=ORDER_TYPE_MARKET):
        """주문 실행"""
        try:
            order = self.client.create_test_order(
                symbol=self.symbol,
                side=side,
                type=order_type,
                quantity=quantity
            )
            return order
        except Exception as e:
            print(f"주문 실패: {e}")
            return None 