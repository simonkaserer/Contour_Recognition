import subprocess
import sys
import cv2
import yaml

from PyQt5 import QtCore, QtGui, QtWidgets
#from QtImageViewer import QtImageViewer

class MainWindow():
    def __init__(self, ContourExtraction,bool):
        super(MainWindow,self).__init__()
        self.filename=''
        self.bufferFilename=''
        if self.filename is not self.bufferFilename:
            self.lineEdit_filename.setText(self.filename)
            self.bufferFilename=self.filename
        
        self.load_items_boxes()
        self.sort_items_boxes()
           

        ContourExtraction.setObjectName("ContourExtraction")
        ContourExtraction.setWindowModality(QtCore.Qt.WindowModal)
        ContourExtraction.resize(1280, 710)
        #ContourExtraction.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        ContourExtraction.setWindowOpacity(1.0)
        ContourExtraction.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.Austria))
        self.centralwidget = QtWidgets.QWidget(ContourExtraction)
        self.centralwidget.setObjectName("centralwidget")

        self.ContourView = QtWidgets.QLabel(self.centralwidget)
        self.ContourView.setGeometry(QtCore.QRect(40, 150, 500, 500))
        self.ContourView.setText("")
        self.ContourView.setTextFormat(QtCore.Qt.AutoText)
        self.ContourView.setPixmap(QtGui.QPixmap("./GUI/CalLeft1.jpg"))
        self.ContourView.setScaledContents(True)
        self.ContourView.setObjectName("ContourView")

        self.lineEdit_Path = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_Path.setGeometry(QtCore.QRect(40, 100, 470, 30))
        self.lineEdit_Path.setObjectName("lineEdit_Path")

        self.Preview = QtWidgets.QLabel(self.centralwidget)
        self.Preview.setGeometry(QtCore.QRect(590, 450, 200, 200))
        self.Preview.setToolTip("Contour Preview")
        self.Preview.setWhatsThis("")
        self.Preview.setText("")
        self.Preview.setPixmap(QtGui.QPixmap("./GUI/CalLeft1.jpg"))
        self.Preview.setScaledContents(True)
        self.Preview.setObjectName("Preview")

        self.button_getContour = QtWidgets.QPushButton(self.centralwidget)
        self.button_getContour.setGeometry(QtCore.QRect(590, 420, 200, 30))
        self.button_getContour.clicked.connect(lambda:update_contour(self,cv2.imread('test.jpg')))
        self.button_getContour.setObjectName("button_getContour")
        
        self.Button_Path = QtWidgets.QToolButton(self.centralwidget)
        self.Button_Path.setGeometry(QtCore.QRect(512, 100, 28, 30))
        self.Button_Path.setObjectName("Button_Path")

        self.Button_savedxf = QtWidgets.QPushButton(self.centralwidget)
        self.Button_savedxf.setGeometry(QtCore.QRect(1200, 590, 60, 60))
        self.Button_savedxf.setObjectName("Button_savedxf")

        self.SliderThresh = QtWidgets.QSlider(self.centralwidget)
        self.SliderThresh.setGeometry(QtCore.QRect(590, 185, 236, 15))
        self.SliderThresh.setOrientation(QtCore.Qt.Horizontal)
        self.SliderThresh.setObjectName("SliderThresh")

        self.Slider_factor = QtWidgets.QSlider(self.centralwidget)
        self.Slider_factor.setGeometry(QtCore.QRect(590, 235, 236, 15))
        self.Slider_factor.setOrientation(QtCore.Qt.Horizontal)
        self.Slider_factor.setObjectName("Slider_factor")

        self.slider3 = QtWidgets.QSlider(self.centralwidget)
        self.slider3.setGeometry(QtCore.QRect(590, 285, 236, 15))
        self.slider3.setOrientation(QtCore.Qt.Horizontal)
        self.slider3.setObjectName("slider3")

        self.comboBox_method = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_method.setGeometry(QtCore.QRect(590, 100, 236, 30))
        self.comboBox_method.setObjectName("comboBox_method")
        self.comboBox_method.addItem('PolyDP')
        self.comboBox_method.addItem('NoApprox')
        self.comboBox_method.addItem('ConvexHull')
        self.comboBox_method.addItem('TehChin')

        self.checkBox_connectpoints = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_connectpoints.setGeometry(QtCore.QRect(590, 320, 141, 28))
        self.checkBox_connectpoints.setObjectName("checkBox_connectpoints")

        self.lineEdit_filename = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_filename.setGeometry(QtCore.QRect(880, 100, 340, 30))
        self.lineEdit_filename.setObjectName("lineEdit_filename")

        self.Button_openKeypad = QtWidgets.QPushButton(self.centralwidget)
        self.Button_openKeypad.setGeometry(QtCore.QRect(960, 20, 131, 30))
        self.Button_openKeypad.setObjectName("Button_openKeypad")
        self.Button_openKeypad.clicked.connect(self.open_keyboard)

        self.Button_closeKeypad = QtWidgets.QPushButton(self.centralwidget)
        self.Button_closeKeypad.setGeometry(QtCore.QRect(1090, 20, 131, 30))
        self.Button_closeKeypad.setObjectName("Button_closeKeypad")
        self.Button_closeKeypad.clicked.connect(self.close_keyboard)

        self.Button_resetFilename = QtWidgets.QPushButton(self.centralwidget)
        self.Button_resetFilename.setGeometry(QtCore.QRect(1050, 620, 141, 30))
        self.Button_resetFilename.setObjectName("Button_resetFilename")
        self.Button_resetFilename.clicked.connect(self.reset_filename)

        self.label_filename = QtWidgets.QLabel(self.centralwidget)
        self.label_filename.setGeometry(QtCore.QRect(880, 72, 68, 22))
        self.label_filename.setObjectName("label_filename")

        self.label_dxfEnding = QtWidgets.QLabel(self.centralwidget)
        self.label_dxfEnding.setGeometry(QtCore.QRect(1222, 100, 68, 30))
        self.label_dxfEnding.setObjectName("label_dxfEnding")

        self.label_slider_thresh = QtWidgets.QLabel(self.centralwidget)
        self.label_slider_thresh.setGeometry(QtCore.QRect(590, 160, 121, 22))
        self.label_slider_thresh.setObjectName("label_slider_thresh")

        self.label_slider_factor = QtWidgets.QLabel(self.centralwidget)
        self.label_slider_factor.setGeometry(QtCore.QRect(590, 210, 121, 22))
        self.label_slider_factor.setObjectName("label_slider_factor")

        self.label_slider3 = QtWidgets.QLabel(self.centralwidget)
        self.label_slider3.setGeometry(QtCore.QRect(590, 260, 68, 22))
        self.label_slider3.setObjectName("label_slider3")
        
        self.label_method = QtWidgets.QLabel(self.centralwidget)
        self.label_method.setGeometry(QtCore.QRect(590, 72, 150, 22))
        self.label_method.setObjectName("label_method")

        self.Label_Path = QtWidgets.QLabel(self.centralwidget)
        self.Label_Path.setGeometry(QtCore.QRect(40, 72, 333, 22))
        self.Label_Path.setObjectName("Label_Path")

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
        self.comboBox_pliers.addItems(self.items['pliers'])
        self.comboBox_pliers.setCurrentText('')
        self.comboBox_pliers.currentTextChanged.connect(self.change_filename)
        self.gridLayout.addWidget(self.comboBox_pliers, 1, 0, 1, 1)

        self.comboBox_sizes = combo(self.widget)
        self.comboBox_sizes.setObjectName("comboBox_sizes")
        self.comboBox_sizes.addItems(self.items['sizes'])
        self.comboBox_sizes.setCurrentText('')
        self.comboBox_sizes.currentTextChanged.connect(self.change_filename)
        self.gridLayout.addWidget(self.comboBox_sizes, 1, 1, 1, 1)

        self.label_measTools = QtWidgets.QLabel(self.widget)
        self.label_measTools.setObjectName("label_measTools")
        self.gridLayout.addWidget(self.label_measTools, 2, 0, 1, 1)

        self.label_numberParts = QtWidgets.QLabel(self.widget)
        self.label_numberParts.setObjectName("label_numberParts")
        self.gridLayout.addWidget(self.label_numberParts, 2, 1, 1, 1)

        self.comboBox_measTools = combo(self.widget)
        self.comboBox_measTools.setObjectName("comboBox_measTools")
        self.comboBox_measTools.addItems(self.items['meas_tools'])
        self.comboBox_measTools.setCurrentText('')
        self.comboBox_measTools.currentTextChanged.connect(self.change_filename)
        self.gridLayout.addWidget(self.comboBox_measTools, 3, 0, 1, 1)

        self.comboBox_numberParts = combo(self.widget)
        self.comboBox_numberParts.setObjectName("comboBox_numberParts")
        self.comboBox_numberParts.addItems(self.items['number_parts'])
        self.comboBox_numberParts.setCurrentText('')
        self.comboBox_numberParts.currentTextChanged.connect(self.change_filename)
        self.gridLayout.addWidget(self.comboBox_numberParts, 3, 1, 1, 1)

        self.label_screwdrivers = QtWidgets.QLabel(self.widget)
        self.label_screwdrivers.setObjectName("label_screwdrivers")
        self.gridLayout.addWidget(self.label_screwdrivers, 4, 0, 1, 1)

        self.label_numbers = QtWidgets.QLabel(self.widget)
        self.label_numbers.setObjectName("label_numbers")
        self.gridLayout.addWidget(self.label_numbers, 4, 1, 1, 1)

        self.comboBox_screwdrivers = combo(self.widget)
        self.comboBox_screwdrivers.setObjectName("comboBox_screwdrivers")
        self.comboBox_screwdrivers.addItems(self.items['screwdrivers'])
        self.comboBox_screwdrivers.setCurrentText('')
        self.comboBox_screwdrivers.currentTextChanged.connect(self.change_filename)
        self.gridLayout.addWidget(self.comboBox_screwdrivers, 5, 0, 1, 1)

        self.comboBox_numbers = combo(self.widget)
        self.comboBox_numbers.setObjectName("comboBox_numbers")
        self.comboBox_numbers.addItems(self.items['numbers'])
        self.comboBox_numbers.setCurrentText('')
        self.comboBox_numbers.currentTextChanged.connect(self.change_filename)
        self.gridLayout.addWidget(self.comboBox_numbers, 5, 1, 1, 1)

        self.label_tools_misc = QtWidgets.QLabel(self.widget)
        self.label_tools_misc.setObjectName("label_tools_misc")
        self.gridLayout.addWidget(self.label_tools_misc, 6, 0, 1, 1)

        self.label_custom = QtWidgets.QLabel(self.widget)
        self.label_custom.setObjectName("label_custom")
        self.gridLayout.addWidget(self.label_custom, 6, 1, 1, 1)

        self.comboBox_tools_misc = combo(self.widget)
        self.comboBox_tools_misc.setObjectName("comboBox_tools_misc")
        self.comboBox_tools_misc.addItems(self.items['tools_misc'])
        self.comboBox_tools_misc.setCurrentText('')
        self.comboBox_tools_misc.currentTextChanged.connect(self.change_filename)
        self.gridLayout.addWidget(self.comboBox_tools_misc, 7, 0, 1, 1)

        self.comboBox_custom = combo(self.widget)
        self.comboBox_custom.setObjectName("comboBox_custom")
        self.comboBox_custom.addItems(self.items['custom'])
        self.comboBox_custom.setCurrentText('')
        self.comboBox_custom.currentTextChanged.connect(self.change_filename)
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
        self.label_newitem.setText(_translate("ContourExtraction", "Put in a text without whitespace here and drag it\n onto the box it should be saved!"))
        self.label_pliers.setText(_translate("ContourExtraction", "Pliers"))
        self.label_sizes.setText(_translate("ContourExtraction", "Sizes"))
        self.label_measTools.setText(_translate("ContourExtraction", "Measurement tools"))
        self.label_numberParts.setText(_translate("ContourExtraction", "Number of parts"))
        self.label_screwdrivers.setText(_translate("ContourExtraction", "Screwdrivers"))
        self.label_numbers.setText(_translate("ContourExtraction", "Numbers"))
        self.label_tools_misc.setText(_translate("ContourExtraction", "Tools"))
        self.label_custom.setText(_translate("ContourExtraction", "Custom"))
    
    def reset_filename(self):
        self.lineEdit_filename.setText('')
        self.filename=''
        boxes=[self.comboBox_pliers,self.comboBox_screwdrivers,self.comboBox_measTools,self.comboBox_tools_misc,self.comboBox_custom,self.comboBox_numberParts,self.comboBox_sizes,self.comboBox_numbers]
        for box in boxes:
            box.setCurrentText('')
        self.save_items_boxes()    
    
    def open_keyboard(self):
        subprocess.call('./open_keyboard.sh')

    def close_keyboard(self):
        subprocess.call('./close_keyboard.sh')

    def update_preview(self,str_image:str):
        self.Preview.setPixmap(QtGui.QPixmap(str_image))

    def change_filename(self):
        self.filename=''
        boxes=[self.comboBox_pliers,self.comboBox_screwdrivers,self.comboBox_measTools,self.comboBox_tools_misc,self.comboBox_custom,self.comboBox_numberParts,self.comboBox_sizes,self.comboBox_numbers]
        for box in boxes:
            str=box.currentText()
            self.filename+=str
        self.lineEdit_filename.setText(self.filename)
    
    def load_items_boxes(self):
        try:
            with open('./ComboBoxItems/items.yaml','r') as f:
                self.items=yaml.safe_load(f)
        except FileNotFoundError as exc:
            self.items={'pliers':['','CombinationPliers','CrimpingPliers'],
            'screwdrivers':['','PoziDriver','PhilipsDriver','FlatDriver'],
            'meas_tools':['','Ruler','Measuringtape','TriangleRuler'],
            'tools_misc':[''],
            'custom':[''],
            'number_parts':['','2pieces','3pieces','4pieces'],
            'sizes':['','Small','Medium','Large'],
            'numbers':['','1','2','3','4','5','6','7','8','9','10']}
    
    def sort_items_boxes(self):
        for i in self.items:
            items=self.items[i]
            items.sort()
            self.items[i]=items
        numbers=self.items['numbers']
        for i in numbers:
            if i.isnumeric() is False:
                numbers.remove(i)
        numbers=sorted(numbers,key=int)
        numbers.insert(0,'')
        self.items['numbers']=numbers

    def save_items_boxes(self):
        self.items['pliers']=[self.comboBox_pliers.itemText(i) for i in range(self.comboBox_pliers.count())]
        self.items['screwdrivers']=[self.comboBox_screwdrivers.itemText(i) for i in range(self.comboBox_screwdrivers.count())]
        self.items['meas_tools']=[self.comboBox_measTools.itemText(i) for i in range(self.comboBox_measTools.count())]
        self.items['tools_misc']=[self.comboBox_tools_misc.itemText(i) for i in range(self.comboBox_tools_misc.count())]
        self.items['custom']=[self.comboBox_custom.itemText(i) for i in range(self.comboBox_custom.count())]
        self.items['number_parts']=[self.comboBox_numberParts.itemText(i) for i in range(self.comboBox_numberParts.count())]
        self.items['sizes']=[self.comboBox_sizes.itemText(i) for i in range(self.comboBox_sizes.count())]
        self.items['numbers']=[self.comboBox_numbers.itemText(i) for i in range(self.comboBox_numbers.count())]   
       
        with open('./ComboBoxItems/items.yaml','w') as f:
            yaml.safe_dump(self.items,f)

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
      
    #   all_items=[self.itemText(i) for i in range(self.count())]
    #     #save all of the items to a individual yaml file per combo box
    #   dict={self.objectName():all_items}
    #   with open('./ComboBoxItems/'+self.objectName()+'.yaml','w') as f:
    #     yaml.safe_dump(dict,f)
      


def update_contour(gui,image):
    #convert the openCv image data into a Qimage
    img = QtGui.QImage(image.data,image.shape[1],image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
    gui.ContourView.setPixmap(QtGui.QPixmap.fromImage(img))    

def main():
    app = QtWidgets.QApplication(sys.argv)
    ContourExtraction = QtWidgets.QMainWindow()
    gui = MainWindow(ContourExtraction,True)
    #gui.setupUi(ContourExtraction)
    ContourExtraction.show()
    #gui.Preview.setPixmap(QtGui.QPixmap('./test.jpg'))
    #image = cv2.imread('test.jpg')
    #update_contour(gui,image)
    sys.exit(app.exec_())
    

if __name__ == '__main__':
    
    main()
    
        
        
