import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMainWindow
from mainUI import Ui_MainWindow
from Train import train
from validate import validate


class Main(QMainWindow, Ui_MainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)

    def train(self):
        if train():
            self.trainResult.setText("训练完成")

    def validate(self):
        filename, _ = QFileDialog.getOpenFileName(self, '打开图片')
        print(filename)
        img = QtGui.QPixmap(filename).scaled(self.pic.width(), self.pic.height())
        self.pic.setPixmap(img)
        self.validateResult.setText(validate(filename))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())
