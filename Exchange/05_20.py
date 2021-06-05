import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import *
import pybithumb

tickers = ["BTC", "ETH", "BCH", "ETC"]
form_class = uic.loadUiType("ui_UPandDOWN.ui")[0]

class MyWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        timer = QTimer(self)
        timer.start(1000)
        timer.timeout.connect(self.timeout)

    def timeout(self):
        print('5 sec!!')
        for i, ticker in enumerate(tickers):
            item = QTableWidgetItem(ticker)
            self.tableWidget.setItem(i, 0, item)
            price, last_ma5, state = self.get_market_info(ticker)
            self.tableWidget.setItem(i, 1, QTableWidgetItem(str(price)))
            self.tableWidget.setItem(i, 2, QTableWidgetItem(str(last_ma5)))
            self.tableWidget.setItem(i, 3, QTableWidgetItem(str(state)))

    def get_market_info(self, ticker):
        df = pybithumb.get_ohlcv(ticker)
        ma5 = df['close'].rolling(window=5).mean()
        last_ma5 = ma5[-2]
        price = pybithumb.get_current_price(ticker)

        state = None
        if price > last_ma5:
            state = 'Up'
        else:
            state = 'Down'

        return price, last_ma5, state

app = QApplication(sys.argv)
win = MyWindow()
win.show()
app.exec_()