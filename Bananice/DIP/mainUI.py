from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QFont


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        MainWindow.setLayoutDirection(QtCore.Qt.LeftToRight)
        MainWindow.setAutoFillBackground(True)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(350, 30, 91, 31))
        font = QtGui.QFont()
        font.setFamily("04b_21")
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label.setObjectName("label")
        self.btn_validate = QtWidgets.QPushButton(self.centralwidget)
        self.btn_validate.setGeometry(QtCore.QRect(430, 310, 81, 31))
        self.btn_validate.setObjectName("btn_validate")
        self.btn_train = QtWidgets.QPushButton(self.centralwidget)
        self.btn_train.setGeometry(QtCore.QRect(300, 310, 81, 31))
        self.btn_train.setObjectName("btn_train")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(270, 380, 71, 31))
        font = QtGui.QFont()
        font.setFamily("04b_21")
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(270, 440, 71, 31))
        font = QtGui.QFont()
        font.setFamily("04b_21")
        font.setPointSize(12)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.pic = QtWidgets.QLabel(self.centralwidget)
        self.pic.setGeometry(QtCore.QRect(260, 80, 261, 191))
        self.pic.setAutoFillBackground(False)
        self.pic.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.pic.setText("")
        self.pic.setObjectName("pic")
        self.trainResult = QtWidgets.QTextEdit(self.centralwidget)
        self.trainResult.setGeometry(QtCore.QRect(360, 380, 151, 31))
        self.trainResult.setObjectName("trainResult")
        self.validateResult = QtWidgets.QTextEdit(self.centralwidget)
        self.validateResult.setGeometry(QtCore.QRect(360, 440, 151, 31))
        self.validateResult.setObjectName("validateResult")
        self.validateResult.setFont(QFont('Arial', 12))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.btn_train.clicked.connect(MainWindow.train)
        self.btn_validate.clicked.connect(MainWindow.validate)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "花卉识别"))
        self.btn_validate.setText(_translate("MainWindow", "识别"))
        self.btn_train.setText(_translate("MainWindow", "训练"))
        self.label_2.setText(_translate("MainWindow", "训练结果"))
        self.label_3.setText(_translate("MainWindow", "识别结果"))
        self.trainResult.setHtml(_translate("MainWindow", "未训练"))
