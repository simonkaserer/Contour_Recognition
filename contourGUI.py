import subprocess
import sys
import cv2
import depthai as dai
from soupsieve import match
import yaml
import Functions
import os
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
#from QtImageViewer import QtImageViewer

class MainWindow():
    def __init__(self, ContourExtraction):
        super(MainWindow,self).__init__()
        self.filename=''
        #self.bufferFilename=''
        self.contour=None
        self.image=None
        self.warped_image=None
        self.cropped_image=None
        self.extraction_image=None
        #The shadowboards are eliminated - only the 550x550mm is valid
        self.scaling=1.0
        

        self.load_items_boxes()
        self.sort_items_boxes()
        self.load_prefs()

        # Load the calibration data
        self.mtx_Rgb=np.load('./CalData/mtx_Rgb.npy')
        self.dist_Rgb=np.load('./CalData/dist_Rgb.npy')
        self.newcameramtx_Rgb=np.load('./CalData/newcameramtx_Rgb.npy')

        ContourExtraction.setObjectName("ContourExtraction")
        ContourExtraction.setWindowModality(QtCore.Qt.WindowModal)
        ContourExtraction.resize(1280, 710)
        ContourExtraction.setWindowOpacity(1.0)
        ContourExtraction.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.Austria))
        self.centralwidget = QtWidgets.QWidget(ContourExtraction)
        self.centralwidget.setObjectName("centralwidget")

        # Set up a timer for a repeated refreshing of the preview
        self.timer=QtCore.QTimer(self.centralwidget)
        self.timer.setInterval(5000)
        self.timer.timeout.connect(self.update_preview)

        self.ContourView = QtWidgets.QLabel(self.centralwidget)
        self.ContourView.setGeometry(QtCore.QRect(40, 150, 500, 500))
        self.ContourView.setText("")
        self.ContourView.setScaledContents(False)
        self.ContourView.setObjectName("ContourView")

        self.Preview = QtWidgets.QLabel(self.centralwidget)
        self.Preview.setGeometry(QtCore.QRect(590, 450, 200, 200))
        self.Preview.setToolTip("Contour Preview")
        self.Preview.setText("")
        self.Preview.setScaledContents(True)
        self.Preview.setObjectName("Preview")

        self.button_getContour = QtWidgets.QPushButton(self.centralwidget)
        self.button_getContour.setGeometry(QtCore.QRect(590, 420, 200, 30))
        self.button_getContour.clicked.connect(self.get_contour)
        self.button_getContour.setObjectName("button_getContour")
        
        self.Button_Path = QtWidgets.QToolButton(self.centralwidget)
        self.Button_Path.setGeometry(QtCore.QRect(512, 100, 28, 30))
        self.Button_Path.clicked.connect(self.open_folder)
        self.Button_Path.setObjectName("Button_Path")

        self.button_savedxf = QtWidgets.QPushButton(self.centralwidget)
        self.button_savedxf.setGeometry(QtCore.QRect(1200, 590, 60, 60))
        self.button_savedxf.clicked.connect(self.save_dxf_button)
        self.button_savedxf.setStyleSheet("background-color:green")
        self.button_savedxf.setObjectName("button_savedxf")

        self.Button_openKeypad = QtWidgets.QPushButton(self.centralwidget)
        self.Button_openKeypad.setGeometry(QtCore.QRect(960, 20, 131, 30))
        self.Button_openKeypad.setObjectName("Button_openKeypad")
        self.Button_openKeypad.clicked.connect(self.open_keyboard)

        self.Button_closeKeypad = QtWidgets.QPushButton(self.centralwidget)
        self.Button_closeKeypad.setGeometry(QtCore.QRect(1090, 20, 130, 30))
        self.Button_closeKeypad.setObjectName("Button_closeKeypad")
        self.Button_closeKeypad.clicked.connect(self.close_keyboard)

        self.Button_resetFilename = QtWidgets.QPushButton(self.centralwidget)
        self.Button_resetFilename.setGeometry(QtCore.QRect(1090, 70, 130, 30))
        self.Button_resetFilename.setObjectName("Button_resetFilename")
        self.Button_resetFilename.clicked.connect(self.reset_filename)

        self.Button_addNewItem= QtWidgets.QPushButton(self.centralwidget)
        self.Button_addNewItem.setGeometry(QtCore.QRect(1180, 160, 30, 30))
        self.Button_addNewItem.setObjectName("Button_addNewItem")
        font=QtGui.QFont()
        font.setPointSize(15)
        self.Button_addNewItem.setFont(font)
        self.Button_addNewItem.setText('+')
        self.Button_addNewItem.clicked.connect(self.save_new_item_dialog)

        self.slider_thresh = QtWidgets.QSlider(self.centralwidget)
        self.slider_thresh.setGeometry(QtCore.QRect(590, 175, 236, 40))
        self.slider_thresh.setOrientation(QtCore.Qt.Horizontal)
        self.slider_thresh.setStyleSheet("""QSlider::handle:horizontal {background-color: #3289a8; border: 1px solid #5c5c5c;  width:20px; height:40px; border-radius:5px;} """)
        self.slider_thresh.setMaximum(254)
        self.slider_thresh.setValue(self.prefs['threshold'])
        self.slider_thresh.valueChanged.connect(self.threshold_changed)
        self.slider_thresh.setObjectName("slider_thresh")

        self.slider_factor = QtWidgets.QSlider(self.centralwidget)
        self.slider_factor.setGeometry(QtCore.QRect(590, 275, 236, 40))
        self.slider_factor.setOrientation(QtCore.Qt.Horizontal)
        self.slider_factor.setValue(int(self.prefs['factor']*10000))
        self.slider_factor.setMaximum(200)
        self.slider_factor.setStyleSheet("""QSlider::handle:horizontal {background-color: #3289a8; border: 1px solid #5c5c5c;  width:20px; height:40px; border-radius:5px;} """)
        self.slider_factor.valueChanged.connect(self.factor_changed)
        self.slider_factor.setObjectName("Slider_factor")

        self.slider_nth_point = QtWidgets.QSlider(self.centralwidget)
        self.slider_nth_point.setGeometry(QtCore.QRect(590, 225, 236, 40))
        self.slider_nth_point.setOrientation(QtCore.Qt.Horizontal)
        self.slider_nth_point.setMinimum(1)
        self.slider_nth_point.setMaximum(50)
        self.slider_nth_point.setValue(self.prefs['nth_point'])
        self.slider_nth_point.setStyleSheet("""QSlider::handle:horizontal {background-color: #3289a8; border: 1px solid #5c5c5c;  width:20px; height:40px; border-radius:5px;} """)
        self.slider_nth_point.valueChanged.connect(self.slider3_changed)
        self.slider_nth_point.setObjectName("slider3")

        self.comboBox_method = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_method.setGeometry(QtCore.QRect(590, 100, 236, 30))
        self.comboBox_method.setObjectName("comboBox_method")
        self.comboBox_method.addItem('PolyDP')
        self.comboBox_method.addItem('NoApprox')
        self.comboBox_method.addItem('ConvexHull')
        self.comboBox_method.addItem('TehChin')
        self.comboBox_method.addItem('CustomApprox')
        self.comboBox_method.currentTextChanged.connect(self.method_changed)

        self.checkBox_connectpoints = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_connectpoints.setGeometry(QtCore.QRect(590, 320, 141, 28))
        self.checkBox_connectpoints.stateChanged.connect(self.connectpoints_changed)
        self.checkBox_connectpoints.setChecked(self.prefs['connectpoints'])
        self.checkBox_connectpoints.setObjectName("checkBox_connectpoints")

        self.lineEdit_filename = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_filename.setGeometry(QtCore.QRect(880, 100, 340, 30))
        self.lineEdit_filename.textChanged.connect(self.filename_manual)
        self.lineEdit_filename.setObjectName("lineEdit_filename")

        self.lineEdit_Path = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_Path.setGeometry(QtCore.QRect(40, 100, 470, 30))
        self.lineEdit_Path.setObjectName("lineEdit_Path")

        self.lineEdit_newItem = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_newItem.setGeometry(QtCore.QRect(880, 160, 290, 30))
        font = QtGui.QFont()
        font.setPointSize(12)
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
        self.label_slider_factor.setGeometry(QtCore.QRect(590, 260, 121, 22))
        self.label_slider_factor.setObjectName("label_slider_factor")

        self.label_slider3 = QtWidgets.QLabel(self.centralwidget)
        self.label_slider3.setGeometry(QtCore.QRect(590, 210, 121, 22))
        self.label_slider3.setObjectName("label_slider3")
        
        self.label_method = QtWidgets.QLabel(self.centralwidget)
        self.label_method.setGeometry(QtCore.QRect(590, 72, 150, 22))
        self.label_method.setObjectName("label_method")

        self.Label_Path = QtWidgets.QLabel(self.centralwidget)
        self.Label_Path.setGeometry(QtCore.QRect(40, 72, 333, 22))
        self.Label_Path.setObjectName("Label_Path")

        

        self.label_newitem = QtWidgets.QLabel(self.centralwidget)
        self.label_newitem.setGeometry(QtCore.QRect(880, 190, 371, 51))
        self.label_newitem.setObjectName("label_newitem")

        # Grid Layout for combo boxes
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(880, 250, 371, 311))
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

        #Check which method is currently set and show/hide the corresponding sliders
        if self.comboBox_method.currentText()=='PolyDP':
            self.slider_factor.show()
            self.label_slider_factor.show()
        else:
            self.slider_factor.hide()
            self.label_slider_factor.hide()
        
        # Deactivate the saving and get contour Button:
        self.button_savedxf.setEnabled(False)
        self.button_getContour.setEnabled(False)

        #Start the timer
        self.timer.start()

        #Start an instance of the worker object to update the preview
        self.worker=UpdatePreview_worker(self.mtx_Rgb,self.dist_Rgb,self.newcameramtx_Rgb,self.prefs['threshold'])
        self.worker.finished.connect(self.worker_finished)

        self.retranslateUi(ContourExtraction)
        QtCore.QMetaObject.connectSlotsByName(ContourExtraction)

    def retranslateUi(self, ContourExtraction):
        _translate = QtCore.QCoreApplication.translate
        ContourExtraction.setWindowTitle(_translate("ContourExtraction", "Contour Extraction"))
        self.Button_Path.setText(_translate("ContourExtraction", "..."))
        self.ContourView.setToolTip(_translate("ContourExtraction", "Contour view"))
        self.Label_Path.setText(_translate("ContourExtraction", "Path to save the files in"))
        self.button_savedxf.setText(_translate("ContourExtraction", "Save\n dxf"))
        self.label_method.setText(_translate("ContourExtraction", "Method"))
        self.button_getContour.setText(_translate("ContourExtraction", "Get Contour"))
        self.label_slider_thresh.setText(_translate("ContourExtraction", "Threshold"))
        self.label_slider_factor.setText(_translate("ContourExtraction", "Factor Epsilon"))
        self.label_slider3.setText(_translate("ContourExtraction", "Every nth point"))
        self.checkBox_connectpoints.setText(_translate("ContourExtraction", "Connect points"))
        self.label_filename.setText(_translate("ContourExtraction", "Filename"))
        self.label_dxfEnding.setText(_translate("ContourExtraction", ".dxf"))
        self.Button_openKeypad.setText(_translate("ContourExtraction", "Open Keypad"))
        self.Button_closeKeypad.setText(_translate("ContourExtraction", "Close Keypad"))
        self.Button_resetFilename.setText(_translate("ContourExtraction", "Reset Filename"))
        self.label_newitem.setText(_translate("ContourExtraction", "Put in a text without whitespace here and drag it\nonto the box it should be saved or press + !"))
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
    def filename_manual(self):
        self.filename=self.lineEdit_filename.text()
        if self.lineEdit_Path.text() != '' and self.filename !='' and self.contour is not None:
            self.button_savedxf.setEnabled(True)
        else:
            self.button_savedxf.setEnabled(False)
    def change_filename(self):
        self.filename=''
        boxes=[self.comboBox_pliers,self.comboBox_screwdrivers,self.comboBox_measTools,self.comboBox_tools_misc,self.comboBox_custom,self.comboBox_numberParts,self.comboBox_sizes,self.comboBox_numbers]
        for box in boxes:
            str=box.currentText()
            self.filename+=str
        self.lineEdit_filename.setText(self.filename) 
        if self.lineEdit_Path.text() != '' and self.filename !='' and self.contour is not None:
            self.button_savedxf.setEnabled(True)
        else:
            self.button_savedxf.setEnabled(False)
    def load_items_boxes(self):
        try:
            with open('items.yaml','r') as f:
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
       
        with open('items.yaml','w') as f:
            yaml.safe_dump(self.items,f)
    def on_button_savePliers(self):
        self.comboBox_pliers.addItem(self.lineEdit_newItem.text())
        self.Dialog.close()
        self.save_items_boxes()
    def on_button_saveScrewdrivers(self):
        self.comboBox_screwdrivers.addItem(self.lineEdit_newItem.text())
        self.Dialog.close()
        self.save_items_boxes()
    def on_button_saveMeasTools(self):
        self.comboBox_measTools.addItem(self.lineEdit_newItem.text())
        self.Dialog.close()
        self.save_items_boxes()
    def on_button_saveMisc(self):
        self.comboBox_tools_misc.addItem(self.lineEdit_newItem.text())
        self.Dialog.close()
        self.save_items_boxes()
    def on_button_saveCustom(self):
        self.comboBox_custom.addItem(self.lineEdit_newItem.text())
        self.Dialog.close()
        self.save_items_boxes()
    def on_button_saveNumberParts(self):
        self.comboBox_pliers.addItem(self.lineEdit_newItem.text())
        self.Dialog.close()
        self.save_items_boxes()
    def on_button_saveSizes(self):
        self.comboBox_sizes.addItem(self.lineEdit_newItem.text())
        self.Dialog.close()
        self.save_items_boxes()
    def on_button_saveNumbers(self):
        self.comboBox_numbers.addItem(self.lineEdit_newItem.text())
        self.Dialog.close()
        self.save_items_boxes()
    def save_new_item_dialog(self):
        if self.lineEdit_newItem.text() != '':
            self.Dialog = QtWidgets.QDialog()
            self.ui = Dialog()
            self.ui.setupUi(self.Dialog)
            self.Dialog.show()
            self.ui.Button_pliers.clicked.connect(self.on_button_savePliers)
            self.ui.Button_screwdrivers.clicked.connect(self.on_button_saveScrewdrivers)
            self.ui.Button_MeasTools.clicked.connect(self.on_button_saveMeasTools)
            self.ui.Butto_misc.clicked.connect(self.on_button_saveMisc)
            self.ui.Button_custom.clicked.connect(self.on_button_saveCustom)
            self.ui.Button_NumberParts.clicked.connect(self.on_button_saveNumberParts)
            self.ui.Button_sizes.clicked.connect(self.on_button_saveSizes)
            self.ui.Button_numbers.clicked.connect(self.on_button_saveNumbers)
    def open_folder(self):
        folderpath=''
        folderpath=QtWidgets.QFileDialog.getExistingDirectory(self.centralwidget,'Select the path to save the contours')
        if folderpath !='':
            self.lineEdit_Path.setText(folderpath)
            if self.lineEdit_Path.text() != '' and self.filename !='' and self.contour is not None:
                self.button_savedxf.setEnabled(True)
            else:
                self.button_savedxf.setEnabled(False)
    def save_dxf_button(self):
        
        

        if self.lineEdit_Path.text() != '' and self.filename !='':
            path_and_filename=self.lineEdit_Path.text()+'/'+self.filename+'.dxf'
            #remove eventual whitespaces in the filename:
            path_and_filename.replace(' ','')
            if self.contour is not None:
                Functions.dxf_exporter(self.contour,path_and_filename,self.scaling)
                success=os.path.exists(path_and_filename)
                if success:
                    dlg=QtWidgets.QMessageBox(self.centralwidget)
                    dlg.setWindowTitle(' ')
                    dlg.setText('Saving file was successful!')
                    dlg.exec()
                else:
                    dlg=QtWidgets.QMessageBox(self.centralwidget)
                    dlg.setWindowTitle(' ')
                    dlg.setText('File could not be saved!')
                    dlg.exec()
            else:
                dlg=QtWidgets.QMessageBox(self.centralwidget)
                dlg.setWindowTitle(' ')
                dlg.setText('No contour available!')
                dlg.exec()
        else:
            dlg=QtWidgets.QMessageBox(self.centralwidget)
            dlg.setWindowTitle(' ')
            dlg.setText('No path or filename selected!')
            dlg.exec()
    def load_prefs(self):
        try:
            with open('prefs.yaml','r') as f:
                self.prefs=yaml.safe_load(f)
        except FileNotFoundError as exc:
            self.prefs={'threshold':150,'factor':0.0005,'nth_point':1,'connectpoints':True}  
    def save_prefs(self):
        self.prefs['threshold']=self.slider_thresh.value()
        self.prefs['factor']=self.slider_factor.value()/10000
        self.prefs['nth_point']=self.slider_nth_point.value()
        self.prefs['connectpoints']=self.checkBox_connectpoints.isChecked()
        with open('prefs.yaml','w') as f:
            yaml.safe_dump(self.prefs,f)
    def closeEvent(self):
        self.save_prefs()
        self.save_items_boxes()
    def threshold_changed(self):
        self.prefs['threshold']=self.slider_thresh.value()
        self.process()
    def factor_changed(self):
        self.prefs['factor']=float(self.slider_factor.value())/10000
        self.process()
    def slider3_changed(self):
        self.prefs['nth_point']=self.slider_nth_point.value()
        self.process()
    def connectpoints_changed(self):
        self.prefs['connectpoints']=self.checkBox_connectpoints.isChecked()
        self.process()
    def method_changed(self):
        if self.comboBox_method.currentText()=='PolyDP':
            self.slider_factor.show()
            self.label_slider_factor.show()
        else:
            self.slider_factor.hide()
            self.label_slider_factor.hide()
        self.process()
    def worker_finished(self):
        self.timer.start()
    def update_preview(self):   
        warped_image,framewidth,frameheight=self.worker.run()
        
        if warped_image is not None:
            self.warped_image=warped_image
            self.framewidth=framewidth
            self.frameheight=frameheight  
            frame=cv2.cvtColor(warped_image,cv2.COLOR_BGR2RGB)
            img = QtGui.QImage(frame,frame.shape[1],frame.shape[0],frame.strides[0],QtGui.QImage.Format_RGB888)
            self.Preview.setPixmap(QtGui.QPixmap.fromImage(img)) 
            self.scaling=np.average([[self.frameheight/550],[self.framewidth/550]]) 

        # Activate the button if a processable image was warped
        if self.warped_image is not None:    
            
            self.button_getContour.setEnabled(True)
            
        else:
            self.button_getContour.setEnabled(False)
    def get_contour(self):
        self.cropped_image=None
        self.extraction_image=self.warped_image
        while self.cropped_image is None:
            if self.extraction_image is not None:
                self.cropped_image,self.toolwidth,self.toolheight,self.tool_pos_x,self.tool_pos_y=Functions.crop_image(self.extraction_image)
        self.process()    
    def process(self):
        
        contour_image=None
        if self.cropped_image is not None:
        #while contour_image is None:
            if self.comboBox_method.currentText() == 'PolyDP':
                self.contour,contour_image=Functions.extraction_polyDP(self.cropped_image,self.prefs['factor'],self.prefs['nth_point'],self.checkBox_connectpoints.isChecked(),self.toolwidth,self.toolheight)
            elif self.comboBox_method.currentText() == 'NoApprox':
                self.contour,contour_image=Functions.extraction_None(self.cropped_image,self.prefs['nth_point'],self.checkBox_connectpoints.isChecked(),self.toolwidth,self.toolheight)
            elif self.comboBox_method.currentText() == 'ConvexHull':
                self.contour,contour_image=Functions.extraction_convexHull(self.cropped_image,self.prefs['nth_point'],self.checkBox_connectpoints.isChecked(),self.toolwidth,self.toolheight)
            elif self.comboBox_method.currentText() == 'TehChin':
                self.contour,contour_image=Functions.extraction_TehChin(self.cropped_image,self.prefs['nth_point'],self.checkBox_connectpoints.isChecked(),self.toolwidth,self.toolheight)
            elif self.comboBox_method.currentText() == 'CustomApprox':
                print('To be implemented!')
                self.contour,contour_image=Functions.extraction_polyDP(self.cropped_image,self.prefs['factor'],self.prefs['nth_point'],self.checkBox_connectpoints.isChecked(),self.toolwidth,self.toolheight)
        if contour_image is not None:    
            frame=cv2.cvtColor(contour_image,cv2.COLOR_BGR2RGB)
            img = QtGui.QImage(frame,frame.shape[1],frame.shape[0],frame.strides[0],QtGui.QImage.Format_RGB888)
            self.ContourView.setPixmap(QtGui.QPixmap.fromImage(img))

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
      
class Dialog(object):
    def __init__(self):
        super(Dialog,self).__init__()
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.setWindowModality(QtCore.Qt.NonModal)
        Dialog.resize(300, 360)
        self.widget = QtWidgets.QWidget(Dialog)
        self.widget.setGeometry(QtCore.QRect(10, 20, 271, 301))
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.Button_pliers = QtWidgets.QPushButton(self.widget)
        self.Button_pliers.setObjectName("Button_pliers")
        self.gridLayout.addWidget(self.Button_pliers, 0, 0, 1, 1)
        self.Button_sizes = QtWidgets.QPushButton(self.widget)
        self.Button_sizes.setObjectName("Button_sizes")
        self.gridLayout.addWidget(self.Button_sizes, 0, 1, 1, 1)
        self.Button_MeasTools = QtWidgets.QPushButton(self.widget)
        self.Button_MeasTools.setObjectName("Button_MeasTools")
        self.gridLayout.addWidget(self.Button_MeasTools, 1, 0, 1, 1)
        self.Button_NumberParts = QtWidgets.QPushButton(self.widget)
        self.Button_NumberParts.setObjectName("Button_NumberParts")
        self.gridLayout.addWidget(self.Button_NumberParts, 1, 1, 1, 1)
        self.Button_screwdrivers = QtWidgets.QPushButton(self.widget)
        self.Button_screwdrivers.setObjectName("Button_screwdrivers")
        self.gridLayout.addWidget(self.Button_screwdrivers, 2, 0, 1, 1)
        self.Button_numbers = QtWidgets.QPushButton(self.widget)
        self.Button_numbers.setObjectName("Button_numbers")
        self.gridLayout.addWidget(self.Button_numbers, 2, 1, 1, 1)
        self.Butto_misc = QtWidgets.QPushButton(self.widget)
        self.Butto_misc.setObjectName("Butto_misc")
        self.gridLayout.addWidget(self.Butto_misc, 3, 0, 1, 1)
        self.Button_custom = QtWidgets.QPushButton(self.widget)
        self.Button_custom.setObjectName("Button_custom")
        
        self.gridLayout.addWidget(self.Button_custom, 3, 1, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Save new item"))
        self.Button_pliers.setText(_translate("Dialog", "Pliers"))
        self.Button_sizes.setText(_translate("Dialog", "Sizes"))
        self.Button_MeasTools.setText(_translate("Dialog", "Measure Tools"))
        self.Button_NumberParts.setText(_translate("Dialog", "Number Parts"))
        self.Button_screwdrivers.setText(_translate("Dialog", "Screwdrivers"))
        self.Button_numbers.setText(_translate("Dialog", "Numbers"))
        self.Butto_misc.setText(_translate("Dialog", "Tools misc"))
        self.Button_custom.setText(_translate("Dialog", "Custom"))
    
class UpdatePreview_worker(QtCore.QThread):
    def __init__(self,mtx_Rgb,dist_Rgb,newcameramtx_Rgb,threshold):
        super().__init__()
        self.mtx=mtx_Rgb
        self.dist=dist_Rgb
        self.newmtx=newcameramtx_Rgb
        self.threshold=threshold
    
    def run(self):
        edgeRgb = edgeRgbQueue.get()
        image=edgeRgb.getFrame()
        image_undistorted=cv2.undistort(image,self.mtx,self.dist,None,self.newmtx)
        # Warp the image
        warped_image,framewidth,frameheigth=Functions.warp_img(image_undistorted,self.threshold,1,False)
        # Return the warped image if a square was found:
        if warped_image is not None:
            if warped_image.shape[0] > 500:
                return warped_image,framewidth,frameheigth
            else:
                return None,None,None
        else:
            return None,None,None
        



    

if __name__ == '__main__':
    sys._excepthook=sys.excepthook

    def exception_hook(exctype,value,traceback):
        sys._excepthook(exctype,value,traceback)
        sys.exit(1)
    
    sys.excepthook=exception_hook

    # Create pipeline
    pipeline = dai.Pipeline()

    # Define sources and outputs
    camRgb = pipeline.create(dai.node.ColorCamera)
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)

    edgeDetectorLeft = pipeline.create(dai.node.EdgeDetector)
    edgeDetectorRight = pipeline.create(dai.node.EdgeDetector)
    edgeDetectorRgb = pipeline.create(dai.node.EdgeDetector)

    xoutEdgeLeft = pipeline.create(dai.node.XLinkOut)
    xoutEdgeRight = pipeline.create(dai.node.XLinkOut)
    xoutEdgeRgb = pipeline.create(dai.node.XLinkOut)
    xinEdgeCfg = pipeline.create(dai.node.XLinkIn)

    edgeLeftStr = "edge left"
    edgeRightStr = "edge right"
    edgeRgbStr = "edge rgb"
    edgeCfgStr = "edge cfg"

    xoutEdgeLeft.setStreamName(edgeLeftStr)
    xoutEdgeRight.setStreamName(edgeRightStr)
    xoutEdgeRgb.setStreamName(edgeRgbStr)
    xinEdgeCfg.setStreamName(edgeCfgStr)

    # Properties
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    edgeDetectorRgb.setMaxOutputFrameSize(camRgb.getVideoWidth() * camRgb.getVideoHeight())

    # Linking
    monoLeft.out.link(edgeDetectorLeft.inputImage)
    monoRight.out.link(edgeDetectorRight.inputImage)
    camRgb.video.link(edgeDetectorRgb.inputImage)

    edgeDetectorLeft.outputImage.link(xoutEdgeLeft.input)
    edgeDetectorRight.outputImage.link(xoutEdgeRight.input)
    edgeDetectorRgb.outputImage.link(xoutEdgeRgb.input)

    xinEdgeCfg.out.link(edgeDetectorLeft.inputConfig)
    xinEdgeCfg.out.link(edgeDetectorRight.inputConfig)
    xinEdgeCfg.out.link(edgeDetectorRgb.inputConfig)

 
    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:
        # Output/input queues                (name,maxSize,blocking)
        edgeLeftQueue = device.getOutputQueue(edgeLeftStr, 1, False)
        edgeRightQueue = device.getOutputQueue(edgeRightStr, 1, False)
        edgeRgbQueue = device.getOutputQueue(edgeRgbStr, 1, False)
        edgeCfgQueue = device.getInputQueue(edgeCfgStr)

        app = QtWidgets.QApplication(sys.argv)
        ContourExtraction = QtWidgets.QMainWindow()
        gui = MainWindow(ContourExtraction)
        ContourExtraction.show()

        
        # These get called in the individual methods of the QT slots
        # edgeLeft = edgeLeftQueue.get()
        # edgeRight = edgeRightQueue.get()
        # edgeRgb = edgeRgbQueue.get()

        # edgeLeftFrame = edgeLeft.getFrame()
        # edgeRightFrame = edgeRight.getFrame()
        # edgeRgbFrame = edgeRgb.getFrame()

        

        # Save the preferences and the items of the ComboBoxes before closing
        app.aboutToQuit.connect(gui.closeEvent)
        
        sys.exit(app.exec_())
            
            
