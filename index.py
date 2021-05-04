import sys
from PyQt5 import uic, QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import pandas as pd
import re  # Expresiones regulares

from sklearn.model_selection import train_test_split as separar
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from statsmodels.tools.eval_measures import rmse
from sklearn import tree, datasets, linear_model
from subprocess import check_call
import matplotlib.pyplot as plt
import numpy as np
import graphviz
import pydot
import os
import time
import subprocess

os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin'


class AppUI(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        uic.loadUi('App.ui', self)
        W = 693
        H = 295
        self.setFixedSize(W, H)
        self.expanded = False
        self.statusBar().setSizeGripEnabled(False)
        self.statusBar().hide()

        self.dataset = None

        self.btn_Aceptar.setEnabled(False)
        self.btn_Aceptar.clicked.connect(self.ReadData)
        self.btn_Examinar.clicked.connect(self.SelectFile)
        self.btn_Salir.clicked.connect(self.close)
        self.btn_Volver.clicked.connect(self.Back)
        self.btn_Volver.setVisible(False)
        self.lbl_dataItems.setVisible(False)
        self.lbl_dataItems.setWordWrap(True)
        self.BackgroundVideo = 'bg_2.gif'

        # Background Animation
        self.MovieLabel = QLabel()
        self.MovieLabel.setScaledContents(True)
        self.MovieLabel.setGeometry(0, 0, W, H)
        self.movie = QMovie(self.BackgroundVideo)
        self.MovieLabel.setMovie(self.movie)
        self.movie.frameChanged.connect(self.repaint)
        self.movie.setScaledSize(QSize(1043, 646))
        self.movie.setSpeed(200)
        self.movie.start()

    def paintEvent(self, event):
        currentFrame = self.movie.currentPixmap()
        frameRect = currentFrame.rect()
        frameRect.moveCenter(self.rect().center())
        if frameRect.intersects(event.rect()):
            painter = QPainter(self)
            painter.drawPixmap(frameRect.left(), frameRect.top(), currentFrame)

    def SelectFile(self):
        fileName = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open file', 'C:\\Users\\sebas\\Desktop\\Proyecto Mate Comp', "Datasets File (*.csv)")
        self.dataset = fileName[0]

        root_array = re.split('/', self.dataset)
        NameOfDataset = root_array[len(root_array) - 1]
        print(NameOfDataset)

        #self.lbl_File.setText(NameOfDataset)
        self.setWindowTitle(NameOfDataset)
        self.btn_Aceptar.setEnabled(True)
        self.btn_Aceptar.setStyleSheet("background-color: #000")
       
    def Back(self):
        self.btn_Aceptar.setVisible(True)
        self.btn_Examinar.setVisible(True)
        self.btn_Salir.setVisible(True)
        self.lbl_Initial.setVisible(True)
        self.btn_Volver.setVisible(False)
        self.lbl_dataItems.setVisible(False)
        self.lbl_dataItems.setText("")

    def ReadData(self):
        self.btn_Aceptar.setVisible(False)
        self.btn_Examinar.setVisible(False)
        self.btn_Salir.setVisible(False)
        self.btn_Volver.setVisible(True)
        self.lbl_Initial.setVisible(False)

        headers = pd.read_csv(self.dataset).columns.tolist()
        df = pd.read_csv(self.dataset)

        LastRow = df.tail(1)

        df = df.head(int(len(df)))


        uniqueVars = []
        for col in df:
            uniqueVars.append(list(df[col].unique()))


        ifString = False

        if df.dtypes[0] != 'int64':
            # STRING TO INTEGERS
            for i, item in enumerate(headers):
                for j, _var in enumerate(uniqueVars):
                    for z, _othervar in enumerate(uniqueVars[j]):
                        df.loc[df[item] == _othervar, item] = int(z+1)

        data = df.iloc[-1].tolist()
        data = data[0:-1]
        df = df.head(int(len(df) - 1))

        #data['output'] = 2 #VALUE RESPONSE

        Y_vars = []

        for i in range(len(df.index)):
            Y_vars.append(df.iloc[i][headers[-1]])


        clf = tree.DecisionTreeClassifier(criterion='entropy')

        df = df.drop(headers[-1], 1)
        depVar = headers[-1]
        headers.remove(headers[-1]) # Borrando el ultimo elemento del header (juego)


        X = df 
        Y = Y_vars

        model = clf.fit(X, Y)

        text_representation = tree.export_text(clf)
        #print(text_representation)

        fig = plt.figure(figsize=(25,20))
        _ = tree.plot_tree(clf, feature_names=headers ,filled=True)
        
        FileOutputFn = "tree.png"

        fig.savefig(FileOutputFn)

        #print(headers)
        #print(data)
        
        dataItems = {}
        for i, item in enumerate(headers):
            dataItems[item] = data[i]
        
        #print(dataItems)
        prediction = clf.predict([data])
        print(prediction[0])
        self.lbl_dataItems.setVisible(True)
        
        self.lbl_dataItems.setText("Con las variables\n" +  str(dataItems) + "\nTenemos como resutado:\n" + str(depVar).capitalize() + ": " + str(prediction[0]))
        os.system(FileOutputFn)

    

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    GUI = AppUI()
    GUI.show()
    sys.exit(app.exec_())
