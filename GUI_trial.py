import sys
import numpy as np
from matplotlib.backends.qt_compat import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

def onclick(event):
    global clicks
    clicks.append(event.xdata)

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self._title = 'Prueba real-time'
        self.setWindowTitle(self._title)

        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        
        dynamic_canvas = FigureCanvas(Figure(figsize=(10, 10)))
        self._dynamic_ax = dynamic_canvas.figure.subplots()
        dynamic_canvas.figure.canvas.mpl_connect('button_press_event', onclick)
        self._dynamic_ax.grid()
        self._timer = dynamic_canvas.new_timer(
            100, [(self._update_window, (), {})])
        self._timer.start()

        button_stop = QtWidgets.QPushButton('Stop', self)
        
        button_stop.clicked.connect(self._timer.stop)

        button_start = QtWidgets.QPushButton('Start', self)
        
        button_start.clicked.connect(self._timer.start)

        self.table_clicks = QtWidgets.QTableWidget(0, 2)
        self.table_clicks.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)

        other_widget = QtWidgets.QLabel("Other widgets", 
            font=QtGui.QFont("Times", 60, QtGui.QFont.Bold), 
            alignment=QtCore.Qt.AlignCenter)
        
        # layouts

        layout = QtWidgets.QGridLayout(self._main)

        layout.addWidget(dynamic_canvas, 0, 0)
        layout.addWidget(self.table_clicks, 0, 1)
        layout.addWidget(other_widget, 1, 0)

        button_layout = QtWidgets.QVBoxLayout()
        button_layout.addWidget(button_stop)
        button_layout.addWidget(button_start)        

        layout.addLayout(button_layout, 1, 1)

        layout.setColumnStretch(0, 2)
        layout.setColumnStretch(1, 1)

    def _update_window(self):
        self._dynamic_ax.clear()
        global x, y1, y2, y3, N, count_iter, last_number_clicks
        x.append(x[count_iter] + 0.01)
        y1.append(np.random.random())
        idx_inf = max([count_iter-N, 0])
        if last_number_clicks < len(clicks):
            for new_click in clicks[last_number_clicks:(len(clicks))]:
                rowPosition = self.table_clicks.rowCount()
                self.table_clicks.insertRow(rowPosition)
                self.table_clicks.setItem(rowPosition,0, QtWidgets.QTableWidgetItem(str(new_click)))
                self.table_clicks.setItem(rowPosition,1, QtWidgets.QTableWidgetItem("Descripcion"))
            last_number_clicks = len(clicks)
        self._dynamic_ax.plot(x[idx_inf:count_iter], y1[idx_inf:count_iter],'-o', color='b')
        count_iter += 1
        self._dynamic_ax.figure.canvas.draw()

if __name__ == "__main__":
    pressed_key = {}
    clicks = []
    last_number_clicks = len(clicks)
    N = 25
    y1 = [np.random.random()]
    x = [0]
    count_iter = 0
    qapp = QtWidgets.QApplication(sys.argv)
    app = ApplicationWindow()
    app.show()
    sys.exit(qapp.exec_())


'''
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QWidget, QAction, QTabWidget,QVBoxLayout,QComboBox,QLabel,QGridLayout,QLineEdit
from PyQt5.QtGui import QIcon

from PyQt5.QtCore import pyqtSlot

class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 tabs - pythonspot.com'
        self.left = 0
        self.top = 0
        self.width = 500
        self.height = 300
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        self.table_widget = MyTableWidget(self)
        self.setCentralWidget(self.table_widget)
        
        self.show()
    
class MyTableWidget(QWidget):
    
    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        tabs = QTabWidget()
        tabs.addTab(self.preLaunch(), "Pre-Launch")
        #tabs.addTab(self.networkTabUI(), "Network")
        self.layout.addWidget(tabs)
        self.tabs.resize(300,200)
        
        # Add tabs
     # Create first tab
    def preLaunch(self):
        tab1 = QWidget()
        layout = QVBoxLayout()
        tab1.layout = QGridLayout()
        tab1.cb  = QComboBox()
        tab1.cb2 = QComboBox()
        tab1.cb3 = QComboBox()
        tab1.cb4 = QComboBox()
        tab1.ln1 = QLineEdit()
        tab1.ln1.setReadOnly(True)
        tab1.ln2 = QLineEdit()
        tab1.ln2.setReadOnly(True)
        tab1.ln3 = QLineEdit()
        tab1.ln3.setReadOnly(True)
        tab1.ln4 = QLineEdit()
        tab1.ln4.setReadOnly(True)
        tab1.label1 = QLabel("Solution Algorthim")
        tab1.cb.addItems(["Simulated Annealing", "Knapsack"])
        tab1.label2 = QLabel("Initial Solution")
        tab1.cb2.addItems(["Nearest Neighbour", "Sweeping"])
        tab1.label3 = QLabel("Scenario")
        tab1.cb3.addItems(["Moabit86", "Moabit220", "Mouint Jou Park"])
        tab1.label4 = QLabel("Number of MUERMELS")
        tab1.cb4.addItems(["1", "2", "3"])
        pushButton1 = QPushButton("Calculate")
        tab1.label5 = QLabel("Total Traveling Distance",)
        tab1.label6 = QLabel("Solution Energy")
        tab1.label7 = QLabel("Solution Time")
        tab1.label8 = QLabel("Number of Dustbins")
        tab1.layout.addWidget(tab1.label1,0,0)
        tab1.layout.addWidget(tab1.label5,0,1)
        tab1.layout.addWidget(tab1.ln1,1,1)
        tab1.layout.addWidget(tab1.cb, 1,0)
        tab1.layout.addWidget(tab1.label2,2,0)
        tab1.layout.addWidget(tab1.label6,2,1)
        tab1.layout.addWidget(tab1.ln2,3,1)
        tab1.layout.addWidget(tab1.cb2, 3,0)
        tab1.layout.addWidget(tab1.label3,4,0)
        tab1.layout.addWidget(tab1.label7,4,1)
        tab1.layout.addWidget(tab1.ln3,5,1)
        tab1.layout.addWidget(tab1.cb3, 5,0)
        tab1.layout.addWidget(tab1.label4,6,0)
        tab1.layout.addWidget(tab1.label8,6,1)
        tab1.layout.addWidget(tab1.ln4,7,1)
        tab1.layout.addWidget(tab1.cb4, 7,0)
        tab1.layout.addWidget(pushButton1,8,0)
        pushButton1.clicked.connect(on_click)
        tab1.setLayout(tab1.layout)
        return tab1
        
        # Add tabs to widget
        #self.layout.addWidget(self.tabs)
        #self.setLayout(self.layout)
        
    @pyqtSlot()
    def on_click(self):
        print("HELOOOO")
        print(self.tab1.cb.currentText())
        print(self.tab1.cb2.currentText())
        print(self.tab1.cb3.currentText())
        print(self.tab1.cb4.currentText())
        self.tab1.ln1.setText("-------")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
'''