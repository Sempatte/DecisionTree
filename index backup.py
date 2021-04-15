import sys
from PyQt5 import uic, QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import pandas as pd
import re # Expresiones regulares

from sklearn.model_selection import train_test_split as separar
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor as Arbol
from statsmodels.tools.eval_measures import rmse
import matplotlib.pyplot as plt
import numpy as np


class AppUI(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        uic.loadUi('App.ui', self)

        self.setFixedSize(284, 315)
        self.expanded = False
        self.statusBar().setSizeGripEnabled(False)
        self.statusBar().hide()

        self.dataset = None

        self.btn_Aceptar.setEnabled(False)
        self.btn_Aceptar.clicked.connect(self.ReadData)
        self.btn_Examinar.clicked.connect(self.SelectFile)


        self.BackgroundVideo = 'bg_2.gif'

        # Background Animation
        self.MovieLabel = QLabel()
        self.MovieLabel.setScaledContents(True)
        self.MovieLabel.setGeometry(0, 0, 284, 315)      
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
        fileName = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', 'C:\\Users\\sebas\\Desktop\\Proyecto Mate Comp',"Datasets File (*.csv)")
        self.dataset = fileName[0]

        root_array = re.split('/', self.dataset)
        NameOfDataset = root_array[len(root_array) - 1]
        print(NameOfDataset)

        self.lbl_File.setText(NameOfDataset)
        self.btn_Aceptar.setEnabled(True)
        self.btn_Aceptar.setStyleSheet("background-color: #000")

    def ReadData(self):
        df = pd.read_csv(self.dataset)
        df.head(10)
        print(df.to_string()) 

        X = df['nivel'].values.reshape(-1,1)
        y = df['salario'].values.reshape(-1,1)

        '''
        c = df['humedad'].values
        d = df['velocidad viento'].values
        e = df['juego'].values
        '''


     
        X_train, X_test, y_train, y_test = separar(X, y, test_size=0.3, random_state=0)
        #print('La forma de X_train es: ', X_train.shape)
        #print('La forma de y_train es: ', y_train.shape)

        
        '''
        escaladorX = MinMaxScaler()
        X = escaladorX.fit_transform(X).reshape(-1,1)

        escaladorY = MinMaxScaler()
        y = escaladorY.fit_transform(y.reshape(-1,1)) # Si ponemos (-1,1) aquí, el SVR se quejará
        '''

        regresor = Arbol(criterion='mse', random_state=0)
        regresor.fit(X, y)
        
        y_fit = regresor.predict(X).reshape(-1,1)
        #y_pred = regresor.predict(6.5).reshape(-1,1) 
        X_grid = np.arange(min(X), max(X), 0.01)
        X_grid = X_grid.reshape(-1, 1)
        y_grid = regresor.predict(X_grid)

        plt.scatter(X, y, color = 'red')
        plt.plot(X_grid, y_grid, color = 'blue')
        plt.title('Regresión Lineal')
        plt.xlabel('Nivel')
        plt.ylabel('Salario')
        plt.show()
        #print('Para un nivel 6.5 cobraría ', y_pred)


        RMSE = np.sum(rmse(y, y_fit))/len(y)
        print('RMSE = ', RMSE)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    GUI = AppUI()
    GUI.show()
    sys.exit(app.exec_())