import subprocess
import sys

from PyQt5 import QtCore, QtGui, QtWidgets

class MainWindow():
    def __init__(self, ContourExtraction,bool):
        super(MainWindow,self).__init__()
#       self.initUI()

#    def initUI(self, ContourExtraction):
        ContourExtraction.setObjectName("ContourExtraction")
        ContourExtraction.setWindowModality(QtCore.Qt.WindowModal)
        ContourExtraction.resize(1280, 710)
        ContourExtraction.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        ContourExtraction.setWindowOpacity(1.0)
        ContourExtraction.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.Austria))
        self.centralwidget = QtWidgets.QWidget(ContourExtraction)
        self.centralwidget.setObjectName("centralwidget")
        self.Button_Path = QtWidgets.QToolButton(self.centralwidget)
        self.Button_Path.setGeometry(QtCore.QRect(512, 100, 28, 30))
        self.Button_Path.setObjectName("Button_Path")
        self.ContourView = QtWidgets.QLabel(self.centralwidget)
        self.ContourView.setGeometry(QtCore.QRect(40, 150, 500, 500))
        self.ContourView.setText("")
        self.ContourView.setTextFormat(QtCore.Qt.AutoText)
        self.ContourView.setPixmap(QtGui.QPixmap("CalLeft1.png"))
        self.ContourView.setScaledContents(True)
        self.ContourView.setObjectName("ContourView")
        self.Label_Path = QtWidgets.QLabel(self.centralwidget)
        self.Label_Path.setGeometry(QtCore.QRect(40, 72, 333, 22))
        self.Label_Path.setObjectName("Label_Path")
        self.lineEdit_Path = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_Path.setGeometry(QtCore.QRect(40, 100, 470, 30))
        self.lineEdit_Path.setObjectName("lineEdit_Path")
        self.Preview = QtWidgets.QLabel(self.centralwidget)
        self.Preview.setGeometry(QtCore.QRect(590, 450, 200, 200))
        self.Preview.setToolTip("Contour Preview")
        self.Preview.setWhatsThis("")
        self.Preview.setText("")
        self.Preview.setPixmap(QtGui.QPixmap("CalLeft1.jpg"))
        self.Preview.setScaledContents(True)
        self.Preview.setObjectName("Preview")
        self.Button_savedxf = QtWidgets.QPushButton(self.centralwidget)
        self.Button_savedxf.setGeometry(QtCore.QRect(1200, 590, 60, 60))
        self.Button_savedxf.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.Button_savedxf.setAutoFillBackground(False)
        self.Button_savedxf.setObjectName("Button_savedxf")
        self.label_method = QtWidgets.QLabel(self.centralwidget)
        self.label_method.setGeometry(QtCore.QRect(590, 72, 150, 22))
        self.label_method.setObjectName("label_method")
        self.SliderThresh = QtWidgets.QSlider(self.centralwidget)
        self.SliderThresh.setGeometry(QtCore.QRect(590, 185, 236, 15))
        self.SliderThresh.setOrientation(QtCore.Qt.Horizontal)
        self.SliderThresh.setObjectName("SliderThresh")
        self.Slider_factor = QtWidgets.QSlider(self.centralwidget)
        self.Slider_factor.setGeometry(QtCore.QRect(590, 235, 236, 15))
        self.Slider_factor.setOrientation(QtCore.Qt.Horizontal)
        self.Slider_factor.setObjectName("Slider_factor")
        self.comboBox_method = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_method.setGeometry(QtCore.QRect(590, 100, 236, 30))
        self.comboBox_method.setObjectName("comboBox_method")
        self.slider3 = QtWidgets.QSlider(self.centralwidget)
        self.slider3.setGeometry(QtCore.QRect(590, 285, 236, 15))
        self.slider3.setOrientation(QtCore.Qt.Horizontal)
        self.slider3.setObjectName("slider3")
        self.button_getContour = QtWidgets.QPushButton(self.centralwidget)
        self.button_getContour.setGeometry(QtCore.QRect(590, 420, 200, 30))
        self.button_getContour.setObjectName("button_getContour")
        self.label_slider_thresh = QtWidgets.QLabel(self.centralwidget)
        self.label_slider_thresh.setGeometry(QtCore.QRect(590, 160, 121, 22))
        self.label_slider_thresh.setObjectName("label_slider_thresh")
        self.label_slider_factor = QtWidgets.QLabel(self.centralwidget)
        self.label_slider_factor.setGeometry(QtCore.QRect(590, 210, 121, 22))
        self.label_slider_factor.setObjectName("label_slider_factor")
        self.label_slider3 = QtWidgets.QLabel(self.centralwidget)
        self.label_slider3.setGeometry(QtCore.QRect(590, 260, 68, 22))
        self.label_slider3.setObjectName("label_slider3")
        self.checkBox_connectpoints = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_connectpoints.setGeometry(QtCore.QRect(590, 320, 141, 28))
        self.checkBox_connectpoints.setObjectName("checkBox_connectpoints")
        self.lineEdit_filename = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_filename.setGeometry(QtCore.QRect(880, 100, 340, 30))
        self.lineEdit_filename.setObjectName("lineEdit_filename")
        self.label_filename = QtWidgets.QLabel(self.centralwidget)
        self.label_filename.setGeometry(QtCore.QRect(880, 72, 68, 22))
        self.label_filename.setObjectName("label_filename")
        self.label_dxfEnding = QtWidgets.QLabel(self.centralwidget)
        self.label_dxfEnding.setGeometry(QtCore.QRect(1222, 100, 68, 30))
        self.label_dxfEnding.setObjectName("label_dxfEnding")
        self.Button_openKeypad = QtWidgets.QPushButton(self.centralwidget)
        self.Button_openKeypad.setGeometry(QtCore.QRect(960, 20, 131, 30))
        self.Button_openKeypad.setObjectName("Button_openKeypad")
        self.Button_openKeypad.clicked.connect(open_keyboard)
        self.Button_closeKeypad = QtWidgets.QPushButton(self.centralwidget)
        self.Button_closeKeypad.setGeometry(QtCore.QRect(1090, 20, 131, 30))
        self.Button_closeKeypad.setObjectName("Button_closeKeypad")
        self.Button_closeKeypad.clicked.connect(close_keyboard)
        self.Button_resetFilename = QtWidgets.QPushButton(self.centralwidget)
        self.Button_resetFilename.setGeometry(QtCore.QRect(1050, 620, 141, 30))
        self.Button_resetFilename.setObjectName("Button_resetFilename")

        self.lineEdit_newItem = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_newItem.setGeometry(QtCore.QRect(880, 140, 290, 50))
        font = QtGui.QFont()
        font.setPointSize(23)
        font.setBold(False)
        font.setWeight(50)
        self.lineEdit_newItem.setFont(font)
        self.lineEdit_newItem.setTabletTracking(True)
        self.lineEdit_newItem.setText("")
        self.lineEdit_newItem.setEchoMode(QtWidgets.QLineEdit.Normal)
        self.lineEdit_newItem.setCursorPosition(0)
        self.lineEdit_newItem.setDragEnabled(True)
        self.lineEdit_newItem.setClearButtonEnabled(False)
        self.lineEdit_newItem.setObjectName("lineEdit_newItem")

        self.label_newitem = QtWidgets.QLabel(self.centralwidget)
        self.label_newitem.setGeometry(QtCore.QRect(880, 190, 371, 51))
        self.label_newitem.setObjectName("label_newitem")

        # Grid Layout for combo boxes
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(880, 240, 371, 311))
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)

        self.gridLayout.setObjectName("gridLayout")
        self.label_pliers = QtWidgets.QLabel(self.widget)
        self.label_pliers.setObjectName("label_pliers")
        self.gridLayout.addWidget(self.label_pliers, 0, 0, 1, 1)

        self.label_sizes = QtWidgets.QLabel(self.widget)
        self.label_sizes.setObjectName("label_sizes")
        self.gridLayout.addWidget(self.label_sizes, 0, 1, 1, 1)

        self.comboBox_pliers = combo(self.widget)
        self.comboBox_pliers.setObjectName("comboBox_pliers")
        self.comboBox_pliers.setAcceptDrops(True)
        self.gridLayout.addWidget(self.comboBox_pliers, 1, 0, 1, 1)

        self.comboBox_sizes = combo(self.widget)
        self.comboBox_sizes.setObjectName("comboBox_sizes")
        self.comboBox_sizes.setAcceptDrops(True)
        self.gridLayout.addWidget(self.comboBox_sizes, 1, 1, 1, 1)

        self.label_measTools = QtWidgets.QLabel(self.widget)
        self.label_measTools.setObjectName("label_measTools")
        self.gridLayout.addWidget(self.label_measTools, 2, 0, 1, 1)

        self.label_numberParts = QtWidgets.QLabel(self.widget)
        self.label_numberParts.setObjectName("label_numberParts")
        self.gridLayout.addWidget(self.label_numberParts, 2, 1, 1, 1)

        self.comboBox_measTools = combo(self.widget)
        self.comboBox_measTools.setObjectName("comboBox_measTools")
        self.comboBox_measTools.setAcceptDrops(True)
        self.gridLayout.addWidget(self.comboBox_measTools, 3, 0, 1, 1)

        self.comboBox_numberParts = combo(self.widget)
        self.comboBox_numberParts.setObjectName("comboBox_numberParts")
        self.comboBox_numberParts.setAcceptDrops(True)
        self.gridLayout.addWidget(self.comboBox_numberParts, 3, 1, 1, 1)

        self.label_screwdrivers = QtWidgets.QLabel(self.widget)
        self.label_screwdrivers.setObjectName("label_screwdrivers")
        self.gridLayout.addWidget(self.label_screwdrivers, 4, 0, 1, 1)

        self.label_numbers = QtWidgets.QLabel(self.widget)
        self.label_numbers.setObjectName("label_numbers")
        self.gridLayout.addWidget(self.label_numbers, 4, 1, 1, 1)

        self.comboBox_screwdrivers = combo(self.widget)
        self.comboBox_screwdrivers.setObjectName("comboBox_screwdrivers")
        self.comboBox_screwdrivers.setAcceptDrops(True)
        self.gridLayout.addWidget(self.comboBox_screwdrivers, 5, 0, 1, 1)

        self.comboBox_numbers = combo(self.widget)
        self.comboBox_numbers.setObjectName("comboBox_numbers")
        self.comboBox_numbers.setAcceptDrops(True)
        self.gridLayout.addWidget(self.comboBox_numbers, 5, 1, 1, 1)

        self.label_tools_misc = QtWidgets.QLabel(self.widget)
        self.label_tools_misc.setObjectName("label_tools_misc")
        self.gridLayout.addWidget(self.label_tools_misc, 6, 0, 1, 1)

        self.label_custom = QtWidgets.QLabel(self.widget)
        self.label_custom.setObjectName("label_custom")
        self.gridLayout.addWidget(self.label_custom, 6, 1, 1, 1)

        self.comboBox_tools_misc = combo(self.widget)
        self.comboBox_tools_misc.setObjectName("comboBox_tools_misc")
        self.comboBox_tools_misc.setAcceptDrops(True)
        self.gridLayout.addWidget(self.comboBox_tools_misc, 7, 0, 1, 1)

        self.comboBox_custom = combo(self.widget)
        self.comboBox_custom.setObjectName("comboBox_custom")
        self.comboBox_custom.setAcceptDrops(True)

        self.gridLayout.addWidget(self.comboBox_custom, 7, 1, 1, 1)

        ContourExtraction.setCentralWidget(self.centralwidget)

        self.retranslateUi(ContourExtraction)
        QtCore.QMetaObject.connectSlotsByName(ContourExtraction)

    def retranslateUi(self, ContourExtraction):
        _translate = QtCore.QCoreApplication.translate
        ContourExtraction.setWindowTitle(_translate("ContourExtraction", "Contour Extraction"))
        self.Button_Path.setText(_translate("ContourExtraction", "..."))
        self.ContourView.setToolTip(_translate("ContourExtraction", "Contour view"))
        self.Label_Path.setText(_translate("ContourExtraction", "Path "))
        self.Button_savedxf.setText(_translate("ContourExtraction", "Save\n dxf"))
        self.label_method.setText(_translate("ContourExtraction", "Method"))
        self.button_getContour.setText(_translate("ContourExtraction", "Get Contour"))
        self.label_slider_thresh.setText(_translate("ContourExtraction", "Threshold"))
        self.label_slider_factor.setText(_translate("ContourExtraction", "Factor Epsilon"))
        self.label_slider3.setText(_translate("ContourExtraction", "Reserve"))
        self.checkBox_connectpoints.setText(_translate("ContourExtraction", "Connect points"))
        self.label_filename.setText(_translate("ContourExtraction", "Filename"))
        self.label_dxfEnding.setText(_translate("ContourExtraction", ".dxf"))
        self.Button_openKeypad.setText(_translate("ContourExtraction", "Open Keypad"))
        self.Button_closeKeypad.setText(_translate("ContourExtraction", "Close Keypad"))
        self.Button_resetFilename.setText(_translate("ContourExtraction", "Reset Filename"))
        self.label_newitem.setText(_translate("ContourExtraction", "Put in a text here and drag it onto the box\nit should be saved!"))
        self.label_pliers.setText(_translate("ContourExtraction", "Pliers"))
        self.label_sizes.setText(_translate("ContourExtraction", "Sizes"))
        self.label_measTools.setText(_translate("ContourExtraction", "Measurement tools"))
        self.label_numberParts.setText(_translate("ContourExtraction", "Number of parts"))
        self.label_screwdrivers.setText(_translate("ContourExtraction", "Screwdrivers"))
        self.label_numbers.setText(_translate("ContourExtraction", "Numbers"))
        self.label_tools_misc.setText(_translate("ContourExtraction", "Tools"))
        self.label_custom.setText(_translate("ContourExtraction", "Custom"))

class combo(QtWidgets.QComboBox):
   def __init__(self, parent):
      super(combo, self).__init__( parent)
      self.setAcceptDrops(True)

   def dragEnterEvent(self, e):
      #print (e)

      if e.mimeData().hasText():
         e.accept()
      else:
         e.ignore()

   def dropEvent(self, e):
      self.addItem(e.mimeData().text())
    
def open_keyboard():
    subprocess.call('./open_keyboard.sh')

def close_keyboard():
    subprocess.call('./close_keyboard.sh')

def main():
    app = QtWidgets.QApplication(sys.argv)
    ContourExtraction = QtWidgets.QMainWindow()
    gui = MainWindow(ContourExtraction,False)
    #gui.setupUi(ContourExtraction)
    ContourExtraction.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
   main()