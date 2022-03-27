# Author: Simon Kaserer
# MCI Bachelor Thesis 2022

#The shadowboards are eliminated - only the 550x550mm lamp perimeter is valid

# Import the needed libraries 
from genericpath import exists
import subprocess
import sys
import cv2
import depthai as dai
import yaml
import Functions
import os
import numpy as np
import time
from PyQt5 import QtCore, QtGui, QtWidgets


class MainWindow():
    # Define the main window class of the GUI. Here the appearance and the methods are defined
    def __init__(self, ContourExtraction):
        # The init function sets up the GUI when the MainWindow class is initialized in the main programme
        super(MainWindow,self).__init__()
        # some variables are defined that the methods can initialize them properly
        self.filename=''
        self.contour=None
        self.image=None
        self.warped_image=None
        self.cropped_image=None
        self.extraction_image=None
        self.scaling_width=1.0
        self.scaling_height=1.0
        self.height=0

        # Load the preferences that are saved with every exit of the program and set the language to the last used one
        self.load_prefs()
        self.language=self.prefs['language']
        # Load the filename parts, sort them and then fill the comboboxes
        self.load_items_boxes()
        self.sort_items_boxes()
        # Load the calibration data out of the CalData folder from the Camera_Calibration.py output
        self.load_cal_data()

        # Set up the GUI with a Main window QTWidget
        ContourExtraction.setObjectName("ContourExtraction")
        ContourExtraction.setWindowModality(QtCore.Qt.WindowModal)
        ContourExtraction.resize(1280, 710)
        ContourExtraction.setWindowOpacity(1.0)
        ContourExtraction.setWindowTitle("Contour Extraction")
        ContourExtraction.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.Austria))
        self.centralwidget = QtWidgets.QWidget(ContourExtraction)
        self.centralwidget.setObjectName("centralwidget")

        # Create the contour view panel. This is a label that gets a picture loaded in with the pixmap function
        self.ContourView = QtWidgets.QLabel(self.centralwidget)
        self.ContourView.setGeometry(QtCore.QRect(40, 150, 500, 500))
        self.ContourView.setText("")
        self.ContourView.setScaledContents(False)
        self.ContourView.setObjectName("ContourView")

        # The preview panel is also a label that gets filled with a picture
        self.Preview = QtWidgets.QLabel(self.centralwidget)
        self.Preview.setGeometry(QtCore.QRect(590, 450, 200, 200))
        self.Preview.setToolTip("Contour Preview")
        self.Preview.setText("")
        self.Preview.setScaledContents(True)
        self.Preview.setObjectName("Preview")

        # Setup of the buttons:
        self.button_getContour = QtWidgets.QPushButton(self.centralwidget)
        self.button_getContour.setGeometry(QtCore.QRect(590, 420, 200, 30))
        self.button_getContour.clicked.connect(self.get_contour)
        self.button_getContour.setObjectName("button_getContour")
        
        self.Button_Path = QtWidgets.QToolButton(self.centralwidget)
        self.Button_Path.setGeometry(QtCore.QRect(1222, 40, 28, 30))
        self.Button_Path.clicked.connect(self.open_folder)
        self.Button_Path.setObjectName("Button_Path")

        self.button_savedxf = QtWidgets.QPushButton(self.centralwidget)
        self.button_savedxf.setGeometry(QtCore.QRect(1170, 590, 80, 60))
        self.button_savedxf.clicked.connect(self.save_dxf_button)
        self.button_savedxf.setStyleSheet("background-color:green")
        self.button_savedxf.setObjectName("button_savedxf")

        self.Button_openKeypad = QtWidgets.QPushButton(self.centralwidget)
        self.Button_openKeypad.setGeometry(QtCore.QRect(40, 20, 150, 30))
        self.Button_openKeypad.setObjectName("Button_openKeypad")
        self.Button_openKeypad.clicked.connect(self.open_keyboard)

        self.Button_closeKeypad = QtWidgets.QPushButton(self.centralwidget)
        self.Button_closeKeypad.setGeometry(QtCore.QRect(190, 20, 150, 30))
        self.Button_closeKeypad.setObjectName("Button_closeKeypad")
        self.Button_closeKeypad.clicked.connect(self.close_keyboard)

        self.Button_resetFilename = QtWidgets.QPushButton(self.centralwidget)
        self.Button_resetFilename.setGeometry(QtCore.QRect(1040, 80, 180, 30))
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

        # Setup of the sliders:
        self.slider_thresh = QtWidgets.QSlider(self.centralwidget)
        self.slider_thresh.setGeometry(QtCore.QRect(590, 175, 236, 40))
        self.slider_thresh.setOrientation(QtCore.Qt.Horizontal)
        self.slider_thresh.setStyleSheet("""QSlider::handle:horizontal {background-color: #3289a8; border: 1px solid #5c5c5c;  width:20px; height:40px; border-radius:5px;} """)
        self.slider_thresh.setMinimum(10) 
        self.slider_thresh.setMaximum(254)
        self.slider_thresh.setValue(self.prefs['threshold'])
        self.slider_thresh.valueChanged.connect(self.threshold_changed)
        self.slider_thresh.setObjectName("slider_thresh")

        self.slider_factor = QtWidgets.QSlider(self.centralwidget)
        self.slider_factor.setGeometry(QtCore.QRect(590, 275, 236, 40))
        self.slider_factor.setOrientation(QtCore.Qt.Horizontal)
        self.slider_factor.setValue(int(self.prefs['factor']*10000))
        self.slider_factor.setMaximum(100)
        self.slider_factor.setStyleSheet("""QSlider::handle:horizontal {background-color: #3289a8; border: 1px solid #5c5c5c;  width:20px; height:40px; border-radius:5px;} """)
        self.slider_factor.valueChanged.connect(self.factor_changed)
        self.slider_factor.setObjectName("Slider_factor")

        self.slider_nth_point = QtWidgets.QSlider(self.centralwidget)
        self.slider_nth_point.setGeometry(QtCore.QRect(590, 225, 236, 40))
        self.slider_nth_point.setOrientation(QtCore.Qt.Horizontal)
        self.slider_nth_point.setMinimum(1)
        self.slider_nth_point.setMaximum(20)
        self.slider_nth_point.setValue(self.prefs['nth_point'])
        self.slider_nth_point.setStyleSheet("""QSlider::handle:horizontal {background-color: #3289a8; border: 1px solid #5c5c5c;  width:20px; height:40px; border-radius:5px;} """)
        self.slider_nth_point.valueChanged.connect(self.slider3_changed)
        self.slider_nth_point.setObjectName("slider3")

        # Setup of the combobox for the method 
        self.comboBox_method = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_method.setGeometry(QtCore.QRect(590, 110, 236, 30))
        self.comboBox_method.setObjectName("comboBox_method")
        self.comboBox_method.addItem('PolyDP')
        self.comboBox_method.addItem('NoApprox')
        self.comboBox_method.addItem('ConvexHull')
        self.comboBox_method.addItem('TehChin')
        self.comboBox_method.addItem('Spline')
        self.comboBox_method.addItem('Spline TehChin')
        self.comboBox_method.setCurrentText(self.prefs['method'])
        self.comboBox_method.currentTextChanged.connect(self.method_changed)

        # Set up the two checkboxes for connecting the points and using the height information in the filename
        self.checkBox_connectpoints = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_connectpoints.setGeometry(QtCore.QRect(590, 320, 141, 28))
        self.checkBox_connectpoints.stateChanged.connect(self.connectpoints_changed)
        self.checkBox_connectpoints.setChecked(self.prefs['connectpoints'])
        self.checkBox_connectpoints.setObjectName("checkBox_connectpoints")

        self.checkBox_height = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_height.setGeometry(QtCore.QRect(590, 360, 250, 28))
        self.checkBox_height.setChecked(self.prefs['save_height'])
        self.checkBox_height.setObjectName("checkBox_height")

        # Set up the text input lines for the filename, the path and the new item
        self.lineEdit_filename = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_filename.setGeometry(QtCore.QRect(880, 110, 340, 30))
        self.lineEdit_filename.textChanged.connect(self.filename_manual)
        self.lineEdit_filename.setObjectName("lineEdit_filename")

        self.lineEdit_Path = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_Path.setGeometry(QtCore.QRect(590, 40, 630, 30)) 
        self.lineEdit_Path.setReadOnly(True) # Set the lineEdit to read only mode to avoid wrong paths typed in via keyboard
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

        # Set up the labels:
        self.label_filename = QtWidgets.QLabel(self.centralwidget)
        self.label_filename.setGeometry(QtCore.QRect(880, 90, 90, 22))
        self.label_filename.setObjectName("label_filename")

        self.label_dxfEnding = QtWidgets.QLabel(self.centralwidget)
        self.label_dxfEnding.setGeometry(QtCore.QRect(1222, 110, 68, 30))
        self.label_dxfEnding.setObjectName("label_dxfEnding")

        self.label_slider_thresh = QtWidgets.QLabel(self.centralwidget)
        self.label_slider_thresh.setGeometry(QtCore.QRect(590, 160, 180, 22))
        self.label_slider_thresh.setObjectName("label_slider_thresh")

        self.label_slider_factor = QtWidgets.QLabel(self.centralwidget)
        self.label_slider_factor.setGeometry(QtCore.QRect(590, 260, 180, 22))
        self.label_slider_factor.setObjectName("label_slider_factor")

        self.label_slider3 = QtWidgets.QLabel(self.centralwidget)
        self.label_slider3.setGeometry(QtCore.QRect(590, 210, 180, 22))
        self.label_slider3.setObjectName("label_slider3")
        
        self.label_method = QtWidgets.QLabel(self.centralwidget)
        self.label_method.setGeometry(QtCore.QRect(590, 85, 150, 22))
        self.label_method.setObjectName("label_method")

        self.Label_Path = QtWidgets.QLabel(self.centralwidget)
        self.Label_Path.setGeometry(QtCore.QRect(590, 20, 333, 22))
        self.Label_Path.setObjectName("Label_Path")

        self.label_newitem = QtWidgets.QLabel(self.centralwidget)
        self.label_newitem.setGeometry(QtCore.QRect(880, 190, 371, 51))
        self.label_newitem.setObjectName("label_newitem")

        self.label_hint = QtWidgets.QLabel(self.centralwidget)
        self.label_hint.setGeometry(QtCore.QRect(40, 80, 500, 30))
        self.label_hint.setFrameShape(QtWidgets.QFrame.Box)
        self.label_hint.setAlignment(QtCore.Qt.AlignCenter)
        self.label_hint.setObjectName("label_hint")

        self.label_position = QtWidgets.QLabel(self.centralwidget)
        self.label_position.setGeometry(QtCore.QRect(40, 115, 500, 31))
        self.label_position.setAlignment(QtCore.Qt.AlignCenter)
        self.label_position.setObjectName("label_position")
        
        self.label_height_value =QtWidgets.QLabel(self.centralwidget)
        self.label_height_value.setGeometry(QtCore.QRect(650,385,141,28))
        self.label_height_value.setText(f'{round(self.height,0)}mm')
        self.label_height_value.setObjectName("label_height")

        self.label_height =QtWidgets.QLabel(self.centralwidget)
        self.label_height.setGeometry(QtCore.QRect(590,385,60,28))
        self.label_height.setObjectName("label_height")

        # The filename comboboxes are placed into a grid layout for a tidy look
        ###############################################################################
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(880, 250, 370, 310))
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
        self.comboBox_pliers.currentTextChanged.connect(self.change_filename)
        self.gridLayout.addWidget(self.comboBox_pliers, 1, 0, 1, 1)

        self.comboBox_sizes = combo(self.widget)
        self.comboBox_sizes.setObjectName("comboBox_sizes")
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
        self.comboBox_measTools.currentTextChanged.connect(self.change_filename)
        self.gridLayout.addWidget(self.comboBox_measTools, 3, 0, 1, 1)

        self.comboBox_numberParts = combo(self.widget)
        self.comboBox_numberParts.setObjectName("comboBox_numberParts")
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
        self.comboBox_screwdrivers.currentTextChanged.connect(self.change_filename)
        self.gridLayout.addWidget(self.comboBox_screwdrivers, 5, 0, 1, 1)

        self.comboBox_numbers = combo(self.widget)
        self.comboBox_numbers.setObjectName("comboBox_numbers")
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
        self.comboBox_tools_misc.currentTextChanged.connect(self.change_filename)
        self.gridLayout.addWidget(self.comboBox_tools_misc, 7, 0, 1, 1)

        self.comboBox_custom = combo(self.widget)
        self.comboBox_custom.setObjectName("comboBox_custom")
        self.comboBox_custom.currentTextChanged.connect(self.change_filename)
        self.gridLayout.addWidget(self.comboBox_custom, 7, 1, 1, 1)

        # Fill the comboboxes with the filename pieces
        self.fill_comboBoxes()
        #################################################################################

        # Setup the Menu bar:#######################################################################
        self.menuBar = QtWidgets.QMenuBar(ContourExtraction)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 1280, 25))
        self.menuBar.setObjectName("menuBar")
        self.menuLanguage = QtWidgets.QMenu(self.menuBar)
        self.menuLanguage.setObjectName("menuLanguage")
        self.menuExtras = QtWidgets.QMenu(self.menuBar)
        self.menuExtras.setObjectName("menuExtras")
        self.menuInfo = QtWidgets.QMenu(self.menuBar)
        self.menuInfo.setObjectName("menuInfo")
        self.actionEnglish = QtWidgets.QAction(ContourExtraction)
        self.actionEnglish.setObjectName("actionEnglish")
        self.actionGerman = QtWidgets.QAction(ContourExtraction)
        self.actionGerman.setObjectName("actionGerman")
        self.actionSave_Metadata = QtWidgets.QAction(ContourExtraction)
        self.actionSave_Metadata.setObjectName("actionSave_Metadata")
        self.actionSave_Contour_Image = QtWidgets.QAction(ContourExtraction)
        self.actionSave_Contour_Image.setObjectName("actionSave_Contour_Image")
        self.actionMethods = QtWidgets.QAction(ContourExtraction)
        self.actionMethods.setObjectName("actionMethods")
        self.actionGeneral = QtWidgets.QAction(ContourExtraction)
        self.actionGeneral.setObjectName("actionGeneral")
        ContourExtraction.setMenuBar(self.menuBar)
        self.menuLanguage.addAction(self.actionEnglish)
        self.menuLanguage.addAction(self.actionGerman)
        self.menuExtras.addAction(self.actionSave_Metadata)
        self.menuExtras.addAction(self.actionSave_Contour_Image)
        self.menuBar.addAction(self.menuLanguage.menuAction())
        self.menuBar.addAction(self.menuExtras.menuAction())
        self.menuBar.addAction(self.menuInfo.menuAction())
        self.menuInfo.addAction(self.actionMethods)
        self.menuInfo.addAction(self.actionGeneral)
        self.actionEnglish.triggered.connect(self.lang_english)
        self.actionGerman.triggered.connect(self.lang_german)
        self.actionSave_Contour_Image.triggered.connect(self.save_img)
        self.actionSave_Metadata.triggered.connect(self.save_meta)
        self.actionMethods.triggered.connect(self.info_methos)
        self.actionGeneral.triggered.connect(self.info_general)
        #################################################################################
       
        ContourExtraction.setCentralWidget(self.centralwidget)
        #Check which method is currently set and show/hide the corresponding sliders
        if self.comboBox_method.currentText()=='PolyDP':
            self.slider_factor.show()
            self.label_slider_factor.show()
        else:
            self.slider_factor.hide()
            self.label_slider_factor.hide()
        # Deactivate the saving- and the get contour- Button:
        self.button_savedxf.setEnabled(False)
        self.button_getContour.setEnabled(False)        
        #Start an instance of the worker object to update the preview
        self.worker=UpdatePreview_worker(self.mtx_Rgb,self.dist_Rgb,self.newcameramtx_Rgb,self.prefs['threshold'])
        self.worker.imageUpdate.connect(self.update_preview)
        self.worker.widthUpdate.connect(self.update_framewidth)
        self.worker.heightUpdate.connect(self.update_frameheight)
        # Set the texts of each labelled element
        self.retranslateUi()
        #Start the worker thread
        self.worker.start()

    def lang_english(self):
        # This method saves the current filename pieces into the current active language and then changes the 
        # language, retranslates the GUI and loads the filename pieces into the comboboxes
        self.save_items_boxes()
        self.language='English'
        self.retranslateUi()
        self.load_items_boxes()
        self.fill_comboBoxes()
    def lang_german(self):
        # This method saves the current filename pieces into the current active language and then changes the 
        # language, retranslates the GUI and loads the filename pieces into the comboboxes
        self.save_items_boxes()
        self.language='German'
        self.retranslateUi()
        self.load_items_boxes()
        self.fill_comboBoxes()
    def retranslateUi(self):
        # This method checks the currently active language and sets the texts of all translated widgets
        # The default language is English, the only other implemented language is German
        if self.language == 'German':
            self.Button_Path.setText( "...")
            self.ContourView.setToolTip( "Konturvorschau")
            self.Label_Path.setText( "Speicherpfad")
            self.button_savedxf.setText( "dxf \n speichern")
            self.label_method.setText( "Methode")
            self.button_getContour.setText( "Kontur anzeigen")
            self.label_slider_thresh.setText( "Grenzwert Binarisierung")
            self.label_slider_factor.setText( "Faktor Epsilon")
            self.label_slider3.setText( "Punktverringerung")
            self.checkBox_connectpoints.setText( "Punkte verbinden")
            self.checkBox_height.setText("Höhe in Dateiname speichern")
            self.label_filename.setText( "Dateiname")
            self.label_dxfEnding.setText( ".dxf")
            self.Button_openKeypad.setText( "Tastatur öffnen")
            self.Button_closeKeypad.setText( "Tastatur schließen")
            self.Button_resetFilename.setText( "Dateiname zurücksetzen")
            self.label_newitem.setText( "Hier einen Textbaustein eingeben und\nauf die gewünschte Liste ziehen oder + drücken!")
            self.label_pliers.setText( "Zangen")
            self.label_sizes.setText( "Größen")
            self.label_measTools.setText( "Messwerkzeuge")
            self.label_numberParts.setText( "Teileanzahl")
            self.label_screwdrivers.setText( "Schraubenzieher")
            self.label_numbers.setText( "Nummern")
            self.label_tools_misc.setText( "Diverse")
            self.label_custom.setText( "Spezial") 
            self.label_hint.setText("Legen Sie das Werkzeug in die Mitte für die besten Ergebnisse")
            self.actionMethods.setText("Methoden")
            self.actionGeneral.setText("Allgemein")
            self.label_height.setText("Höhe:")
        else:
            self.Button_Path.setText( "...")
            self.ContourView.setToolTip( "Contour view")
            self.Label_Path.setText( "Path to save the files in")
            self.button_savedxf.setText( "Save\n dxf")
            self.label_method.setText( "Method")
            self.button_getContour.setText( "Get Contour")
            self.label_slider_thresh.setText( "Threshold")
            self.label_slider_factor.setText( "Factor Epsilon")
            self.label_slider3.setText( "Every nth point")
            self.checkBox_connectpoints.setText( "Connect points")
            self.checkBox_height.setText("Save height in filename")
            self.label_filename.setText( "Filename")
            self.label_dxfEnding.setText( ".dxf")
            self.Button_openKeypad.setText( "Open Keypad")
            self.Button_closeKeypad.setText( "Close Keypad")
            self.Button_resetFilename.setText( "Reset Filename")
            self.label_newitem.setText( "Put in a text without whitespace here and drag it\nonto the box it should be saved or press + !")
            self.label_pliers.setText( "Pliers")
            self.label_sizes.setText( "Sizes")
            self.label_measTools.setText( "Measurement tools")
            self.label_numberParts.setText( "Number of parts")
            self.label_screwdrivers.setText( "Screwdrivers")
            self.label_numbers.setText( "Numbers")
            self.label_tools_misc.setText( "Tools")
            self.label_custom.setText( "Custom") 
            self.label_hint.setText("For best results place the tool in the middle of the plate")
            self.actionMethods.setText("Methods")
            self.actionGeneral.setText("General")
            self.label_height.setText("Height:")
        # These texts are not affected by the language change:
        self.menuLanguage.setTitle( "Language")
        self.actionEnglish.setText( "English")
        self.actionGerman.setText( "German")
        self.menuExtras.setTitle("Extras")
        self.actionSave_Metadata.setText("Save Metadata")
        self.actionSave_Contour_Image.setText("Save Contour Image")
        self.menuInfo.setTitle("Info")
        
    def save_meta(self): # This is a method for testing purposes where the current settings and some other informations are saved under the 
        # selected path and filename but in a .yaml format
        if self.lineEdit_Path.text() != '' and self.filename !='' and self.contour is not None:
            self.prefs['threshold']=self.slider_thresh.value()
            self.prefs['factor']=self.slider_factor.value()/10000
            self.prefs['nth_point']=self.slider_nth_point.value()
            self.prefs['connectpoints']=self.checkBox_connectpoints.isChecked()
            self.prefs['framewidth']=self.framewidth
            self.prefs['frameheight']=self.frameheight
            self.prefs['toolwidth']=self.toolwidth
            self.prefs['toolheight']=self.toolheight
            self.prefs['x']=self.tool_pos_x
            self.prefs['y']=self.tool_pos_y
            self.prefs['scaling_width']=self.scaling_width
            self.prefs['scaling_height']=self.scaling_height
            path=self.lineEdit_Path.text()+'/'+self.filename+'.yaml'
            with open(path,'w') as f:
                yaml.safe_dump(self.prefs,f)
    def save_img(self): # This method saves the images of the left and right mono camera. They can be used for further computings 
        if self.lineEdit_Path.text() != '' and self.filename !='' and self.contour is not None:
            path=self.lineEdit_Path.text()+'/'+self.filename+'.jpg'
            pathleft=self.lineEdit_Path.text()+'/'+self.filename+'Left.jpg'
            pathright=self.lineEdit_Path.text()+'/'+self.filename+'Right.jpg'
            cv2.imwrite(path,self.cropped_image)
            edgeLeft = edgeLeftQueue.get()
            image=edgeLeft.getFrame()
            cv2.imwrite(pathleft,image)
            edgeRight = edgeRightQueue.get()
            image=edgeRight.getFrame()
            cv2.imwrite(pathright,image)   
    def open_keyboard(self): # Opens the display keyboard through a bash script that stores the PID into a file
        subprocess.call('./open_keyboard.sh')
    def close_keyboard(self): # Closes the display keyboard if a instance of it runs 
        subprocess.call('./close_keyboard.sh')
    def filename_manual(self): # Checks if a path is selected and a contour is detected when a 
        # filename is typed in the line edit directly and enables the button for saving the contour
        self.filename=self.lineEdit_filename.text()
        if self.lineEdit_Path.text() != '' and self.filename !='' and self.contour is not None:
            self.button_savedxf.setEnabled(True)
        else:
            self.button_savedxf.setEnabled(False)
    def change_filename(self): # Checks all of the comboboxes for a selected filename piece and cumulates them
        self.filename=''
        # All of the box objects are put into an array to have easy access to them in a for-loop
        boxes=[self.comboBox_pliers,self.comboBox_screwdrivers,self.comboBox_measTools,self.comboBox_tools_misc,self.comboBox_custom,self.comboBox_numberParts,self.comboBox_sizes,self.comboBox_numbers]
        for box in boxes:
            # Every string is added to the filename. If nothing is selected the string is empty and doesn't affect the filename
            str=box.currentText()
            self.filename+=str
        self.lineEdit_filename.setText(self.filename) 
        # If the contour extists, the path is selected and the filename is not empty the button for saving the contour is enabled
        if self.lineEdit_Path.text() != '' and self.filename !='' and self.contour is not None:
            self.button_savedxf.setEnabled(True)
        else:
            self.button_savedxf.setEnabled(False)
    def reset_filename(self): # Deletes the current filename and sets the text of all comboboxes to an empty string
        self.lineEdit_filename.setText('')
        self.filename=''
        # All of the box objects are put into an array to ease the resetting
        boxes=[self.comboBox_pliers,self.comboBox_screwdrivers,self.comboBox_measTools,self.comboBox_tools_misc,self.comboBox_custom,self.comboBox_numberParts,self.comboBox_sizes,self.comboBox_numbers]
        for box in boxes:
            box.setCurrentText('')
    def load_cal_data(self): # This method tries to load the numpy arrays from the Camera_Calibration.py output in the CalData folder
        try:
            self.mtx_Rgb=np.load('./CalData/mtx_Rgb.npy')
            self.dist_Rgb=np.load('./CalData/dist_Rgb.npy')
            self.newcameramtx_Rgb=np.load('./CalData/newcameramtx_Rgb.npy')
            self.mtx_right=np.load('./CalData/mtx_right.npy')
            self.dist_right=np.load('./CalData/dist_right.npy')
            self.newcameramtx_right=np.load('./CalData/newcameramtx_right.npy')
            self.mtx_left=np.load('./CalData/mtx_left.npy')
            self.dist_left=np.load('./CalData/dist_left.npy')
            self.newcameramtx_left=np.load('./CalData/newcameramtx_left.npy')
            self.stereoMapL_x=np.load('./CalData/stereoMapL_x.npy')
            self.stereoMapL_y=np.load('./CalData/stereoMapL_y.npy')
            self.stereoMapR_x=np.load('./CalData/stereoMapR_x.npy')
            self.stereoMapR_y=np.load('./CalData/stereoMapR_y.npy')
        except FileNotFoundError as exc: # If the data can't be loaded a Message Box appears and tells the user to 
            # run the Camera_Calibration.py program in order to calibrate the cameras. This prevents the main program from starting
            msg=QtWidgets.QMessageBox()
            msg.setWindowTitle('Error')
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.setText('Calibration Data not found - please run CameraCalibration.py through the Terminal inside the workspace of the contour extraction program.')
            msg.exec_()
            msg.buttonClicked.connect(quit())
    def load_items_boxes(self): # Tries to load the filename pieces from the items_language.yaml files
        # both language items get loaded. If they are not found a default set of items is loaded.
        try:
            with open('items_german.yaml','r') as f:
                self.items_german=yaml.safe_load(f)
        except FileNotFoundError as exc:
            self.items_german={
            'Zangen':['','Kombizange','Crimpzange'],
            'Schraubenzieher':['','Pozi','Philips','Schlitz'],
            'Messwerkzeuge':['','Geodreieck','Rollmaßband','Lineal'],
            'Diverse':[''],
            'Spezial':[''],
            'Teileanzahl':['','2teilig','3teilig','4teilig'],
            'Groessen':['','Klein','Mittel','Groß'],
            'Nummern':['','1','2','3','4','5','6','7','8','9','10']}  
    
        try:
            with open('items_english.yaml','r') as f:
                self.items_english=yaml.safe_load(f)
        except FileNotFoundError as exc:
            self.items_english={
            'pliers':['','CombinationPliers','CrimpingPliers'],
            'screwdrivers':['','PoziDriver','PhilipsDriver','FlatDriver'],
            'meas_tools':['','Ruler','Measuringtape','TriangleRuler'],
            'tools_misc':[''],
            'custom':[''],
            'number_parts':['','2pieces','3pieces','4pieces'],
            'sizes':['','Small','Medium','Large'],
            'numbers':['','1','2','3','4','5','6','7','8','9','10']} 
    def sort_items_boxes(self): # Sorts the items in a alphabetical way. The numbers are extracted
        # and every non-numeric item gets deleted to ensure that the sorting with numeric key is done properly
        # The empty string '' is inserted in the first position that the combobox has the empty string as default item
        for i in self.items_german:
            items=self.items_german[i]
            items.sort()
            self.items_german[i]=items
        numbers=self.items_german['Nummern']
        for i in numbers:
            if i.isnumeric() is False:
                numbers.remove(i)
        numbers=sorted(numbers,key=int)
        numbers.insert(0,'')
        self.items_german['Nummern']=numbers
        for i in self.items_english:
            items=self.items_english[i]
            items.sort()
            self.items_english[i]=items
        numbers=self.items_english['numbers']
        for i in numbers:
            if i.isnumeric() is False:
                numbers.remove(i)
        numbers=sorted(numbers,key=int)
        numbers.insert(0,'')
        self.items_english['numbers']=numbers
    def save_items_boxes(self): # Saves the items inside the comboboxes of the current language when called.
        if self.language=='German':
            self.items_german['Zangen']=[self.comboBox_pliers.itemText(i) for i in range(self.comboBox_pliers.count())]
            self.items_german['Schraubenzieher']=[self.comboBox_screwdrivers.itemText(i) for i in range(self.comboBox_screwdrivers.count())]
            self.items_german['Messwerkzeuge']=[self.comboBox_measTools.itemText(i) for i in range(self.comboBox_measTools.count())]
            self.items_german['Diverse']=[self.comboBox_tools_misc.itemText(i) for i in range(self.comboBox_tools_misc.count())]
            self.items_german['Spezial']=[self.comboBox_custom.itemText(i) for i in range(self.comboBox_custom.count())]
            self.items_german['Teileanzahl']=[self.comboBox_numberParts.itemText(i) for i in range(self.comboBox_numberParts.count())]
            self.items_german['Groessen']=[self.comboBox_sizes.itemText(i) for i in range(self.comboBox_sizes.count())]
            self.items_german['Nummern']=[self.comboBox_numbers.itemText(i) for i in range(self.comboBox_numbers.count())]   
       
            with open('items_german.yaml','w') as f:
                yaml.safe_dump(self.items_german,f)
        else:
            self.items_english['pliers']=[self.comboBox_pliers.itemText(i) for i in range(self.comboBox_pliers.count())]
            self.items_english['screwdrivers']=[self.comboBox_screwdrivers.itemText(i) for i in range(self.comboBox_screwdrivers.count())]
            self.items_english['meas_tools']=[self.comboBox_measTools.itemText(i) for i in range(self.comboBox_measTools.count())]
            self.items_english['tools_misc']=[self.comboBox_tools_misc.itemText(i) for i in range(self.comboBox_tools_misc.count())]
            self.items_english['custom']=[self.comboBox_custom.itemText(i) for i in range(self.comboBox_custom.count())]
            self.items_english['number_parts']=[self.comboBox_numberParts.itemText(i) for i in range(self.comboBox_numberParts.count())]
            self.items_english['sizes']=[self.comboBox_sizes.itemText(i) for i in range(self.comboBox_sizes.count())]
            self.items_english['numbers']=[self.comboBox_numbers.itemText(i) for i in range(self.comboBox_numbers.count())]   
        
            with open('items_english.yaml','w') as f:
                yaml.safe_dump(self.items_english,f)
    def fill_comboBoxes(self): # First clear all the current items of the comboboxes, adds the loaded filename pieces into each combobox and sets the text as an empty string.
        # The instances are written in an array to index them in the for loop:
        boxes=[self.comboBox_pliers,self.comboBox_screwdrivers,self.comboBox_measTools,self.comboBox_tools_misc,self.comboBox_custom,self.comboBox_numberParts,self.comboBox_sizes,self.comboBox_numbers]
        if self.language=='German':
            for box in boxes:
                box.clear()
            self.comboBox_pliers.addItems(self.items_german['Zangen'])
            self.comboBox_pliers.setCurrentText('')
            self.comboBox_sizes.addItems(self.items_german['Groessen'])
            self.comboBox_sizes.setCurrentText('')
            self.comboBox_measTools.addItems(self.items_german['Messwerkzeuge'])
            self.comboBox_measTools.setCurrentText('')
            self.comboBox_numberParts.addItems(self.items_german['Teileanzahl'])
            self.comboBox_numberParts.setCurrentText('')
            self.comboBox_screwdrivers.addItems(self.items_german['Schraubenzieher'])
            self.comboBox_screwdrivers.setCurrentText('')
            self.comboBox_numbers.addItems(self.items_german['Nummern'])
            self.comboBox_numbers.setCurrentText('')
            self.comboBox_tools_misc.addItems(self.items_german['Diverse'])
            self.comboBox_tools_misc.setCurrentText('')
            self.comboBox_custom.addItems(self.items_german['Spezial'])
            self.comboBox_custom.setCurrentText('')
        else:
            for box in boxes:
                box.clear()
            self.comboBox_pliers.addItems(self.items_english['pliers'])
            self.comboBox_pliers.setCurrentText('')
            self.comboBox_sizes.addItems(self.items_english['sizes'])
            self.comboBox_sizes.setCurrentText('')
            self.comboBox_measTools.addItems(self.items_english['meas_tools'])
            self.comboBox_measTools.setCurrentText('')
            self.comboBox_numberParts.addItems(self.items_english['number_parts'])
            self.comboBox_numberParts.setCurrentText('')
            self.comboBox_screwdrivers.addItems(self.items_english['screwdrivers'])
            self.comboBox_screwdrivers.setCurrentText('')
            self.comboBox_numbers.addItems(self.items_english['numbers'])
            self.comboBox_numbers.setCurrentText('')
            self.comboBox_tools_misc.addItems(self.items_english['tools_misc'])
            self.comboBox_tools_misc.setCurrentText('')
            self.comboBox_custom.addItems(self.items_english['custom'])
            self.comboBox_custom.setCurrentText('')
    def on_button_savePliers(self): # Function for adding a filename piece to the chosen combobox when a button in the dialog window is pressed
        self.comboBox_pliers.addItem(self.lineEdit_newItem.text())
        self.Dialog.close()
        self.save_items_boxes()
    def on_button_saveScrewdrivers(self): # Function for adding a filename piece to the chosen combobox when a button in the dialog window is pressed
        self.comboBox_screwdrivers.addItem(self.lineEdit_newItem.text())
        self.Dialog.close()
        self.save_items_boxes()
    def on_button_saveMeasTools(self): # Function for adding a filename piece to the chosen combobox when a button in the dialog window is pressed
        self.comboBox_measTools.addItem(self.lineEdit_newItem.text())
        self.Dialog.close()
        self.save_items_boxes()
    def on_button_saveMisc(self): # Function for adding a filename piece to the chosen combobox when a button in the dialog window is pressed
        self.comboBox_tools_misc.addItem(self.lineEdit_newItem.text())
        self.Dialog.close()
        self.save_items_boxes()
    def on_button_saveCustom(self): # Function for adding a filename piece to the chosen combobox when a button in the dialog window is pressed
        self.comboBox_custom.addItem(self.lineEdit_newItem.text())
        self.Dialog.close()
        self.save_items_boxes()
    def on_button_saveNumberParts(self): # Function for adding a filename piece to the chosen combobox when a button in the dialog window is pressed
        self.comboBox_pliers.addItem(self.lineEdit_newItem.text())
        self.Dialog.close()
        self.save_items_boxes()
    def on_button_saveSizes(self): # Function for adding a filename piece to the chosen combobox when a button in the dialog window is pressed
        self.comboBox_sizes.addItem(self.lineEdit_newItem.text())
        self.Dialog.close()
        self.save_items_boxes()
    def on_button_saveNumbers(self): # Function for adding a filename piece to the chosen combobox when a button in the dialog window is pressed
        self.comboBox_numbers.addItem(self.lineEdit_newItem.text())
        self.Dialog.close()
        self.save_items_boxes()
    def save_new_item_dialog(self): # Creates an instance of the Dialog class if the text in the input line is not empty
        # This class has 8 Buttons that save the filename piece into the chosen combobox
        if self.lineEdit_newItem.text() != '':
            self.Dialog = QtWidgets.QDialog()
            self.ui = Dialog()
            self.ui.setupUi(self.Dialog,self.language)
            self.Dialog.show()
            self.ui.Button_pliers.clicked.connect(self.on_button_savePliers)
            self.ui.Button_screwdrivers.clicked.connect(self.on_button_saveScrewdrivers)
            self.ui.Button_MeasTools.clicked.connect(self.on_button_saveMeasTools)
            self.ui.Butto_misc.clicked.connect(self.on_button_saveMisc)
            self.ui.Button_custom.clicked.connect(self.on_button_saveCustom)
            self.ui.Button_NumberParts.clicked.connect(self.on_button_saveNumberParts)
            self.ui.Button_sizes.clicked.connect(self.on_button_saveSizes)
            self.ui.Button_numbers.clicked.connect(self.on_button_saveNumbers)
    def open_folder(self): # Initiates a file dialog class from PyQT5 for chosing a path.
        # This function limits the selection to folders or directories only
        folderpath=''
        if self.language=='German':
            open_str='Zielordner auswählen'
        else:
            open_str='Select the path to save the contours'
        folderpath=QtWidgets.QFileDialog.getExistingDirectory(self.centralwidget,open_str)
        # If a path is selected it is shown in the text input line
        if folderpath !='':
            self.lineEdit_Path.setText(folderpath)
            if self.lineEdit_Path.text() != '' and self.filename !='' and self.contour is not None:
                self.button_savedxf.setEnabled(True)
            else:
                self.button_savedxf.setEnabled(False)
    def save_dxf_button(self): # This function cumulates the filename with the absolute path and adds the height information if the checkbox is checked.
        if self.lineEdit_Path.text() != '' and self.filename !='':
            if self.checkBox_height.isChecked():
                path_and_filename=self.lineEdit_Path.text()+'/'+self.filename+f'{round(self.height,0)}mm.dxf'
            else:
                path_and_filename=self.lineEdit_Path.text()+'/'+self.filename+'.dxf'
            # Remove eventual whitespaces in the filename:
            path_and_filename.replace(' ','')
            # The dxf_exporter function is called and after a check if the file exists, a message window appears with the confirmation
            # If something goes wrong while saving or the contour, the path or the filename are missing then a error message appears.
            if self.contour is not None:
                Functions.dxf_exporter(self.contour,path_and_filename,self.scaling_width,self.scaling_height)
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
    def load_prefs(self): # This function loads the preferences, if the file can't be found a set of default values are loaded.
        try:
            with open('prefs.yaml','r') as f:
                self.prefs=yaml.safe_load(f)
        except FileNotFoundError as exc:
            self.prefs={'threshold':150,'factor':0.0005,'nth_point':1,'connectpoints':True,'language':'English','method':'Spline','save_height':True}  
    def save_prefs(self): # The values of the preferences to be stored are loaded into the prefs-variable and then are saved as .yaml file
        self.prefs['threshold']=self.slider_thresh.value()
        self.prefs['factor']=self.slider_factor.value()/10000
        self.prefs['nth_point']=self.slider_nth_point.value()
        self.prefs['connectpoints']=self.checkBox_connectpoints.isChecked()
        self.prefs['language']=self.language
        self.prefs['method']=self.comboBox_method.currentText()
        self.prefs['save_height']=self.checkBox_height.isChecked()
        # the with statement prevents the file from staying opened if a exception occurs during the saving process
        with open('prefs.yaml','w') as f:
            yaml.safe_dump(self.prefs,f)
    def threshold_changed(self): # writes the new value to the preferences and starts the process again
        self.prefs['threshold']=self.slider_thresh.value()
        # Update the new threshold value in the worker thread
        self.worker.update_threshold(self.prefs['threshold'])
    def factor_changed(self): # writes the new value to the preferences and starts the process again
        self.prefs['factor']=float(self.slider_factor.value())/10000
        self.process()
    def slider3_changed(self): # writes the new value to the preferences and starts the process again
        self.prefs['nth_point']=self.slider_nth_point.value()
        self.process()
    def connectpoints_changed(self): # Updates the status of the checkbox in the preferences and starts the process again
        self.prefs['connectpoints']=self.checkBox_connectpoints.isChecked()
        self.process()
    def method_changed(self): # Checks which method is selected and hides or shows the assigned sliders and checkboxes and starts the process again.
        if self.comboBox_method.currentText()=='PolyDP':
            self.slider_factor.show()
            self.label_slider_factor.show()
        else:
            self.slider_factor.hide()
            self.label_slider_factor.hide()
        if self.comboBox_method.currentText()=='Spline' or self.comboBox_method.currentText()=='Spline TehChin':
            self.checkBox_connectpoints.hide()
        else:
            self.checkBox_connectpoints.show()
        
        self.process()

    def update_frameheight(self,height): # This function is called when the worker class in the seperate thread emits the values of the frameheight
        self.frameheight=height
    def update_framewidth(self,width): # This function is called when the worker class in the seperate thread emits the values of the framewidth
        self.framewidth=width
    def update_preview(self,warped_image): # When the perimeter of the lamp is found, the worker class emits the image. This function checks the size of 
        # the image and sets the contour preview to the given image with the pixmap routing. Here the image is converted to a Qimage format and then to a
        # Pixmap format. Also the scaling factor is calculated and saved in the according variables.
        if warped_image is not None:
            if warped_image.shape[0]>500:
                self.warped_image=warped_image
                frame=cv2.cvtColor(warped_image,cv2.COLOR_BGR2RGB)
                img = QtGui.QImage(frame,frame.shape[1],frame.shape[0],frame.strides[0],QtGui.QImage.Format_RGB888)
                self.Preview.setPixmap(QtGui.QPixmap.fromImage(img)) 
                self.scaling_height=float(self.frameheight/550)
                self.scaling_width=float(self.framewidth/550)
        # Activate the button if a processable image was warped
        if self.warped_image is not None:      
            self.button_getContour.setEnabled(True)
        else:
            self.button_getContour.setEnabled(False)
    def get_contour(self): # This method extracts the contour out of the warped image
        # To get the images for the height-function the frames of the left and right camera queue are loaded
        edgeLeft=edgeLeftQueue.get()
        edgeRight=edgeRightQueue.get()
        img_left=edgeLeft.getFrame()
        img_right=edgeRight.getFrame()
        # Undistort and rectify the images
        img_left=cv2.remap(img_left,self.stereoMapL_x,self.stereoMapL_y,cv2.INTER_LANCZOS4,cv2.BORDER_CONSTANT,0)
        img_right=cv2.remap(img_right,self.stereoMapR_x,self.stereoMapR_y,cv2.INTER_LANCZOS4,cv2.BORDER_CONSTANT,0)
        # The toolheigt function is called
        Functions.toolheight(img_left,img_right)
        # The cropping function is called until a tool is found and then cropped. With this cropped image the process is started
        self.cropped_image=None
        self.extraction_image=self.warped_image
        while self.cropped_image is None:
            if self.extraction_image is not None:
                self.cropped_image,self.toolwidth,self.toolheight,self.tool_pos_x,self.tool_pos_y=Functions.crop_image(self.extraction_image)
        self.process()    
    def process(self): # This method chooses the called extraction function according to the selected method.
        # The needed parameters are provided to the function
        contour_image=None
        if self.cropped_image is not None:
            if self.comboBox_method.currentText() == 'PolyDP':
                self.contour,contour_image=Functions.extraction_polyDP(self.cropped_image,self.prefs['factor'],self.prefs['nth_point'],self.checkBox_connectpoints.isChecked(),self.toolwidth,self.toolheight)
            elif self.comboBox_method.currentText() == 'NoApprox':
                self.contour,contour_image=Functions.extraction_None(self.cropped_image,self.prefs['nth_point'],self.checkBox_connectpoints.isChecked(),self.toolwidth,self.toolheight)
            elif self.comboBox_method.currentText() == 'ConvexHull':
                self.contour,contour_image=Functions.extraction_convexHull(self.cropped_image,self.prefs['nth_point'],self.checkBox_connectpoints.isChecked(),self.toolwidth,self.toolheight)
            elif self.comboBox_method.currentText() == 'TehChin':
                self.contour,contour_image=Functions.extraction_TehChin(self.cropped_image,self.prefs['nth_point'],self.checkBox_connectpoints.isChecked(),self.toolwidth,self.toolheight)
            elif self.comboBox_method.currentText() == 'Spline':
                self.contour,contour_image=Functions.extraction_spline(self.cropped_image,self.prefs['nth_point'],self.checkBox_connectpoints.isChecked(),self.toolwidth,self.toolheight)
            elif self.comboBox_method.currentText() == 'Spline TehChin':
                self.contour,contour_image=Functions.extraction_spline_tehChin(self.cropped_image,self.prefs['nth_point'],self.checkBox_connectpoints.isChecked(),self.toolwidth,self.toolheight)
        # If a contour is found, it is showed on the big contour view panel
        if contour_image is not None:    
            frame=cv2.cvtColor(contour_image,cv2.COLOR_BGR2RGB)
            img = QtGui.QImage(frame,frame.shape[1],frame.shape[0],frame.strides[0],QtGui.QImage.Format_RGB888)
            self.ContourView.setPixmap(QtGui.QPixmap.fromImage(img))
            # Check if the tool is near the middle of the board and
            # give an information that the tool center is not in the middle:
            move_str=''
            if self.tool_pos_x < self.framewidth/2-30:
                if self.language=='German':
                    move_str+="Nach rechts verschieben. "
                else:
                    move_str+="Move towards the right. "
            elif self.tool_pos_x > self.framewidth/2+30:
                if self.language=='German':
                    move_str+="Nach links verschieben. "
                else:
                    move_str+="Move towards the left. "
            if self.tool_pos_y < self.frameheight/2-30:
                if self.language=='German':
                    move_str+="Nach unten verschieben."
                else:
                    move_str+="Move towards the bottom."
            elif self.tool_pos_y > self.frameheight/2+30:
                if self.language=='German':
                    move_str+="Nach oben verschieben."
                else:
                    move_str+="Move towards the top."
            else:
                move_str=''
            self.label_position.setText(move_str)
    def info_methos(self): # Provides the user with information regarding the used methods when clicking on the info menu bar
        dlg=QtWidgets.QMessageBox(self.centralwidget)
        dlg.setWindowTitle('Info')
        if self.language=='German':
            dlg.setText('Methods:\nPolyDP - Diese Funktion verwendet den Douglas-Peucker-Algorithmus um eine Kurve durch eine Kurve mit weniger Punkten auszudrücken. Der Faktor bestimmt um wieviel die neue Kurve vereinfacht werden darf.\n\n'
            'NoApprox - Gibt sämtliche als Konturpunkte identifizierten Pixel zurück.\n\n'
            'ConvexHull - Extrahiert die Hülle des Werkzeugs. Kann bei kleinen Werkzeugen oder Teilen nützlich sein.\n\n'
            'TehChin - Verwendet die Teh Chin Approximation um die Kontur zu finden. Hier werden Bereiche mit großer Richtungsänderung mit vielen Punkten aufgelöst und Bereiche mit wenig Änderung mit wenigen Punkten.\n\n'
            'Spline - Diese Funktion verwendet alle Punkte der Kontur und verbindet sie mit einem Spline.\n\n'
            'Spline TehChin - Diese Funktion verwendet die Teh Chin Approximation und verbindet die Konturpunkte durch einen Spline.')
        else:
            dlg.setText('Methods:\nPolyDP - This function uses the Douglas-Peucker algorithm to simplify a curve with fewer points. The parameter factor determines how much the new curve can be simplified.\n\n'
            'NoApprox - Returns every found pixel of the contour\n\n'
            'ConvexHull - Finds the hull of the tool. Can be used for small tools or things.\n\n'
            'TehChin - Uses the Teh Chin approximation to find the contour. This algorithm varies the number of found edgepoints according to the change of the direction\n\n'
            'Spline - This function finds the edge with no approximation and connects the contour points with a spline.\n\n'
            'Spline TehChin - This function uses the Teh Chin approximation and connects the contour points with a spline.')
        dlg.exec()
    def info_general(self): # Provides the user with information regarding the structure and the workflow of the program
        dlg=QtWidgets.QMessageBox(self.centralwidget)
        dlg.setWindowTitle('Info')
        if self.language=='German':
            dlg.setText('Dieses Programm verwendet die Bilder der OAK-D Kamera und extrahiert die Kontur eines Werkzeuges. Um die besten Ergebnisse zu erzielen sollte das Werkzeug möglichst nahe der Mitte positioniert werden.'
                        'Sobald der Rahmen der beleuchteten Fläche erkannt wird, kann die Kontur extrahiert werden. Dazu wird der Knopf Kontur anzeigen gedrückt.'
                        'Wird eine Kontur erkannt, so wird sie angezeigt. \nÜber die Schieberegler kann der Grenzwert der Binarisierung eingestellt werden: Dieser ermöglicht das anpassen an durchsichtige oder helle Werkzeuge.\n'
                        'Mit der Punkteverringerung kann die Anzahl der verwendeten Punkte verringert werden. Dadurch verringert sich aber auch die Genauigkeit.\n'
                        '\nWird nun ein Speicherpfad ausgewählt und ein Dateiname vergeben, kann die Kontur im .dxf-Format gespeichert werden'
                        'Die Auswahlboxen beinhalten Vorschläge für die Namensgebung und damit kann ein Dateiname zusammengesetzt werden. Zum Beispiel: KombizangeKlein3 durch auswählen der jeweiligen Bausteine.'
                        'Auch kann der Dateiname direkt mit der Tastatur/Bildschirmtastatur eingegeben werden. Sollte ein neuer Textbaustein gespeichert werden, kann dieser in die dafür vorgesehene Zeile geschrieben werden und per Drag-and-Drop in die jeweilige Box'
                        'hinzugefügt werden. Auch durch das betätigen des + Knopfes kann der Textbaustein zur gewünschten Gruppe hinzugefügt werden.\n'
                        'Sollten Einträge gelöscht werden, kann dies nur über die \'items_german.yaml\'-Datei erfolgen.')
        else:
            dlg.setText('This program uses the images from the OAK-D camera and extracts the contour of a tool. For best results, the tool should be positioned as close to the center as possible.'
                        'Once the frame of the illuminated surface is detected, the contour can be extracted. This is done by pressing the Get Contour button.'
                        'If a contour is detected, it will be displayed. \nThe sliders can be used to set the threshold of the binarization: This allows to adapt to transparent or light tools.\n'
                        'With the \'nth point\' slider the number of used points can be reduced. However, this also reduces the accuracy.'
                        '\nWhen a save path is selected and a file name is assigned, the contour can be saved in .dxf format'
                        'The selection boxes contain suggestions for naming and thus a file name can be composed. For example: CombinationPlierSmall3 by selecting the respective blocks.'
                        'Also, the file name can be entered directly with the keyboard/screen keypad. If a new text block should be saved, it can be written in the line provided and dragged and dropped into the respective box'
                        'to be added. The text module can also be added to the desired group by pressing the + button.'
                        'Should entries be deleted, this can only be done via the \'items_english.yaml\' file.')
        # After setting the text the dialog window is executed
        dlg.exec()         

    def closeEvent(self): # stops the worker thread and saves the preferences and the filename pieces
        self.worker.stop()
        self.save_prefs()
        self.save_items_boxes()

class combo(QtWidgets.QComboBox): # Class definition of the combobox dialog for saving new filename pieces
   def __init__(self, parent): # the initialization is inherited and the AcceptDrops-flag is activated
      super(combo, self).__init__( parent)
      self.setAcceptDrops(True)

   def dragEnterEvent(self, e): # If data is dragged onto the combobox it will accept the data if it is text
      if e.mimeData().hasText():
         e.accept()
      else:
         e.ignore()

   def dropEvent(self, e): # If the data is accepted, the text is added as item in the combobox
      self.addItem(e.mimeData().text())      
class Dialog(object): # Definition of the dialog class for the filename piece saving
    def __init__(self):
        super(Dialog,self).__init__()
        
    def setupUi(self, Dialog,language): # Here the appearance of the dialog is created
        self.language=language
        Dialog.setObjectName("Dialog")
        Dialog.setWindowModality(QtCore.Qt.NonModal)
        Dialog.resize(300, 360)
        # The title is named according to the language chosen
        if self.language =='German':
            Dialog.setWindowTitle("Textbaustein speichern")
        else:
            Dialog.setWindowTitle("Save new item")
        self.widget = QtWidgets.QWidget(Dialog)
        self.widget.setGeometry(QtCore.QRect(10, 20, 271, 301))
        self.widget.setObjectName("widget")
        # The buttons are added to a grid layout to ensure a tidy appearance
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
        # The buttons are labelled in the chosen language in this seperate function
        self.retranslateUi(self.language)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, language): # Labels the button according to the passed language
        if language == 'German':
            self.Button_pliers.setText("Zangen")
            self.Button_sizes.setText("Größen")
            self.Button_MeasTools.setText("Messwerkzeuge")
            self.Button_NumberParts.setText("Teileanzahl")
            self.Button_screwdrivers.setText("Schraubenzieher")
            self.Button_numbers.setText("Nummern")
            self.Butto_misc.setText("Diverse")
            self.Button_custom.setText("Spezial")
        else:
            self.Button_pliers.setText("Pliers")
            self.Button_sizes.setText("Sizes")
            self.Button_MeasTools.setText("Measure Tools")
            self.Button_NumberParts.setText("Number Parts")
            self.Button_screwdrivers.setText("Screwdrivers")
            self.Button_numbers.setText("Numbers")
            self.Butto_misc.setText("Tools misc")
            self.Button_custom.setText("Custom")
class UpdatePreview_worker(QtCore.QThread): # Class definition of the threaded worker class
    # Define the signals that are emitted during the run of the worker thread
    imageUpdate=QtCore.pyqtSignal(np.ndarray) 
    widthUpdate=QtCore.pyqtSignal(int)
    heightUpdate=QtCore.pyqtSignal(int)
    def __init__(self,mtx_Rgb,dist_Rgb,newcameramtx_Rgb,threshold): # Saves the passed values into variables during the initialization
        super().__init__()
        self.mtx=mtx_Rgb
        self.dist=dist_Rgb
        self.newmtx=newcameramtx_Rgb
        self.threshold=threshold
    
    def stop(self): # Stops the while loop and quits the worker thread
        self.ThreadActive=False
        self.quit()
    
    def update_threshold(self,threshold): # Updates the threshold value in the worker thread
        self.threshold=threshold

    def run(self): # This sets up a while loop that runs until the ThreadActive boolean is disabled
        self.ThreadActive=True
        while self.ThreadActive:
            # Get the latest data in the data queue of the OAK-D 
            edgeRgb = edgeRgbQueue.get()
            image=edgeRgb.getFrame()
            # Also get the latest data of the mono cameras to ensure the queue holds the lates picture
            edgeLeftQueue.get()
            edgeRightQueue.get()
            
            image_undistorted=cv2.undistort(image,self.mtx,self.dist,None,self.newmtx)
            # Warp the image and emit the values and the image to be processed in the GUI class
            warped_image,framewidth,frameheigth,_=Functions.warp_img(image_undistorted,self.threshold,1,False)
            if warped_image is not None:
                self.widthUpdate.emit(framewidth)
                self.heightUpdate.emit(frameheigth)
                self.imageUpdate.emit(warped_image)
            time.sleep(0.5)
        
  

if __name__ == '__main__': # Main program
    sys._excepthook=sys.excepthook
    # Define a new exception hook:
    def exception_hook(exctype,value,traceback):
        sys._excepthook(exctype,value,traceback)
        sys.exit(1)
    sys.excepthook=exception_hook

    # Create the pipeline for the OAK-D
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

    # Properties of the cameras
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    edgeDetectorRgb.setMaxOutputFrameSize(camRgb.getVideoWidth() * camRgb.getVideoHeight())

    # Linking of the OAK-D nodes
    monoLeft.out.link(edgeDetectorLeft.inputImage)
    monoRight.out.link(edgeDetectorRight.inputImage)
    camRgb.video.link(edgeDetectorRgb.inputImage)

    edgeDetectorLeft.outputImage.link(xoutEdgeLeft.input)
    edgeDetectorRight.outputImage.link(xoutEdgeRight.input)
    edgeDetectorRgb.outputImage.link(xoutEdgeRgb.input)

    xinEdgeCfg.out.link(edgeDetectorLeft.inputConfig)
    xinEdgeCfg.out.link(edgeDetectorRight.inputConfig)
    xinEdgeCfg.out.link(edgeDetectorRgb.inputConfig)

 
    # Connect to the device and start the pipeline
    with dai.Device(pipeline) as device:
        # Output/input queues                (name,maxSize,blocking)
        edgeLeftQueue = device.getOutputQueue(edgeLeftStr, 1, False)
        edgeRightQueue = device.getOutputQueue(edgeRightStr, 1, False)
        edgeRgbQueue = device.getOutputQueue(edgeRgbStr, 1, False)
        edgeCfgQueue = device.getInputQueue(edgeCfgStr)
        # Set an instance of the GUI
        app = QtWidgets.QApplication(sys.argv)
        ContourExtraction = QtWidgets.QMainWindow()
        gui = MainWindow(ContourExtraction)
        ContourExtraction.show()

        # Save the preferences and the items of the ComboBoxes before closing
        app.aboutToQuit.connect(gui.closeEvent)
        # Start the execution of the GUI
        sys.exit(app.exec_())
            
            
