import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import *

from_class = uic.loadUiType("mywindow_cobit.ui")[0]

class MyWindow(QMainWindow, from_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.btn_clicked)
        self.setWindowIcon(QIcon('Bitcoin-BTC-icon.png'))

        # self.setGeometry(100,200,300,400)
        # self.setWindowTitle('PyQt')

        # btn = QPushButton("버튼", self)
        # btn.move(10,10)
        # btn.clicked.connect(self.btn_clicked)

    def btn_clicked(self):
        print('버튼 클릭')

app = QApplication(sys.argv)
window = MyWindow()
window.show()
app.exec_()