import pybithumb

# df = pybithumb.get_ohlcv("BTC")
# ma5 = df['close'].rolling(window=5).mean()
# last_ma5 = ma5[-2]
# price = pybithumb.get_current_price("BTC")

# if price > last_ma5:
#     print('UP', price,last_ma5)
# else:
#     print('DOWN', price,last_ma5)

# print(ma5)
# print(ma5[4])

def bull_market(ticker):
    df = pybithumb.get_ohlcv(ticker)
    ma5 = df['close'].rolling(window=5).mean()
    print = pybithumb.get_current_price(ticker)
    last_ma5 = ma5[-2]

    if price > last_ma5:
        return True
    else:
        return False

is_bull= bull_market("DOGE")
if is_bull:
    print("UP")