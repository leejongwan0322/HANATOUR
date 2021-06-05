import pybithumb
import time
import datetime

tickers = pybithumb.get_tickers()
# print(tickers)
# print(len(tickers))
# price = pybithumb.get_current_price("BTC")
# print(price)

detail = pybithumb.get_market_detail("DOGE")
print(detail)

orderbook = pybithumb.get_orderbook("DOGE")
# print(orderbook)
ms = int(orderbook["timestamp"])
dt = datetime.datetime.fromtimestamp(ms/1000)
print(dt)

# for k in orderbook:
#     print(k)

# while True:
#     price = pybithumb.get_current_price("DOGE")
#     print(price)
#     time.sleep(5)
