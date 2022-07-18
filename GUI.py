import sys
#import io
from PyQt5.QtWidgets import QTextEdit, QTableWidgetItem,QMainWindow, QApplication, QPushButton, QWidget, QAction, QTabWidget,QVBoxLayout,QComboBox,QLabel,QGridLayout,QLineEdit
#from PyQt5.QtGui import QIcon

from PyQt5.QtCore import pyqtSlot
#from PyQt5.QtWebEngineWidgets import QWebEngineView 
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt	
#import ks_trial 

class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'MURMEL Rute Planning'
        self.left = 0
        self.top = 0
        self.width = 870
        self.height = 500
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
        
        # Add tabs
        self.tabs.addTab(self.tab1,"Pre-Launch")
        self.tabs.addTab(self.tab2,"Active")
        #self.tabs.addTab(self.tab3,"Post-Launch")
        
        # Create 1st tab
        self.tab1.layout = QGridLayout(self.tab1)
        self.tab1.cb  = QComboBox(self)
        self.tab1.cb2 = QComboBox(self)
        self.tab1.cb3 = QComboBox(self)
        self.tab1.cb4 = QComboBox(self)
        self.tab1.ln1 = QLineEdit(self)
        self.tab1.ln1.setReadOnly(True)
        self.tab1.ln2 = QLineEdit(self)
        self.tab1.ln2.setReadOnly(True)
        self.tab1.ln3 = QLineEdit(self)
        self.tab1.ln3.setReadOnly(True)
        self.tab1.ln4 = QLineEdit(self)
        self.tab1.ln4.setReadOnly(True)
        self.tab1.ln5 = QTextEdit(self)
        self.tab1.ln5.setReadOnly(True)
        self.tab1.label1 = QLabel("Solution Algorthim",self)
        self.tab1.cb.addItems(["Simulated Annealing", "Knapsack"])
        self.tab1.label2 = QLabel("Initial Solution",self)
        self.tab1.cb2.addItems(["Nearest Neighbour", "Sweeping"])
        self.tab1.label3 = QLabel("Scenario",self)
        self.tab1.cb3.addItems(["Moabit86", "Moabit220", "Mouint Jou Park"])
        self.tab1.label4 = QLabel("Number of MUERMELS",self)
        self.tab1.cb4.addItems(["1", "2", "3"])
        self.tab1.pushButton1 = QPushButton("Calculate")
        self.tab1.label5 = QLabel("Total Traveling Distance",self)
        self.tab1.label6 = QLabel("Solution Energy",self)
        self.tab1.label7 = QLabel("Solution Time",self)
        self.tab1.label8 = QLabel("Number of Dustbins",self)
        self.tab1.layout.addWidget(self.tab1.label1,0,0)
        self.tab1.layout.addWidget(self.tab1.label5,0,1)
        self.tab1.layout.addWidget(self.tab1.ln1,1,1)
        self.tab1.layout.addWidget(self.tab1.cb, 1,0)
        self.tab1.layout.addWidget(self.tab1.label2,2,0)
        self.tab1.layout.addWidget(self.tab1.label6,2,1)
        self.tab1.layout.addWidget(self.tab1.ln2,3,1)
        self.tab1.layout.addWidget(self.tab1.cb2, 3,0)
        self.tab1.layout.addWidget(self.tab1.label3,4,0)
        self.tab1.layout.addWidget(self.tab1.label7,4,1)
        self.tab1.layout.addWidget(self.tab1.ln3,5,1)
        self.tab1.layout.addWidget(self.tab1.cb3, 5,0)
        self.tab1.layout.addWidget(self.tab1.label4,6,0)
        self.tab1.layout.addWidget(self.tab1.label8,6,1)
        self.tab1.layout.addWidget(self.tab1.ln4,7,1)
        self.tab1.layout.addWidget(self.tab1.cb4, 7,0)
        self.tab1.layout.addWidget(self.tab1.pushButton1,8,0)
        self.tab1.layout.addWidget(self.tab1.ln5,9,0,2,2)
        dynamic_canvas = FigureCanvas(Figure(figsize=(9, 5)))
        self.tab1._dynamic_ax = dynamic_canvas.figure.subplots()
        self.tab1._dynamic_ax.tick_params(labelsize=6)
        self.tab1._dynamic_ax.grid()
        self.tab1._dynamic_ax.plot(x,y,'-o', color='black')
        #self.tab1._timer = dynamic_canvas.new_timer(100, [(self._update_window, (), {})])
        #self.tab1._timer.start()
        self.tab1.pushButton1.clicked.connect(self.on_click)
        self.tab1.layout.addWidget(dynamic_canvas, 0, 3,15,4)
        self.tab1.setLayout(self.tab1.layout)
        
        # Create 2nd tab
        self.tab2.layout = QGridLayout(self.tab2)
        self.tab2.ln1 = QLineEdit(self)
        self.tab2.ln1.setReadOnly(True)
        self.tab2.ln2 = QLineEdit(self)
        self.tab2.ln2.setReadOnly(True)
        self.tab2.ln3 = QLineEdit(self)
        self.tab2.ln3.setReadOnly(True)
        self.tab2.ln4 = QLineEdit(self)
        self.tab2.ln4.setReadOnly(True)
        self.tab2.ln5 = QLineEdit(self)
        self.tab2.ln5.setReadOnly(True)
        self.tab2.ln6 = QLineEdit(self)
        self.tab2.ln6.setReadOnly(True)
        self.tab2.ln7 = QLineEdit(self)
        self.tab2.ln7.setReadOnly(True)
        self.tab2.ln8 = QLineEdit(self)
        self.tab2.ln8.setReadOnly(True)
        self.tab2.ln9 = QLineEdit(self)
        self.tab2.ln9.setReadOnly(True)
        self.tab2.ln10 = QTextEdit(self)
        self.tab2.ln10.setReadOnly(True)
        self.tab2.pushButton1 = QPushButton("Start")
        self.tab2.pushButton2 = QPushButton("Stop")
        self.tab2.label1 = QLabel("Total Time passed",self)
        self.tab2.label2 = QLabel("Total Traveled Distance MMs",self)
        self.tab2.label3 = QLabel("Total Energy Consumed MMs",self)
        self.tab2.label4 = QLabel("Total Traveled Distance MS",self)
        self.tab2.label9 = QLabel("Total Energy Consumed MS",self)
        self.tab2.label5 = QLabel("Number of Visited Dustbins",self)
        self.tab2.label6 = QLabel("Battery Changes",self)
        self.tab2.label7 = QLabel("Total Trash Pick-Ups",self)
        self.tab2.label8 = QLabel("Battery Changes",self)
        self.tab2.layout.addWidget(self.tab2.label1,0,0)
        self.tab2.layout.addWidget(self.tab2.label5,0,1)
        self.tab2.layout.addWidget(self.tab2.ln1,1,1)
        self.tab2.layout.addWidget(self.tab2.ln5, 1,0)
        self.tab2.layout.addWidget(self.tab2.label2,2,0)
        self.tab2.layout.addWidget(self.tab2.label6,2,1)
        self.tab2.layout.addWidget(self.tab2.ln2,3,1)
        self.tab2.layout.addWidget(self.tab2.ln6, 3,0)
        self.tab2.layout.addWidget(self.tab2.label3,4,0)
        self.tab2.layout.addWidget(self.tab2.label7,4,1)
        self.tab2.layout.addWidget(self.tab2.ln3,5,1)
        self.tab2.layout.addWidget(self.tab2.ln7, 5,0)
        self.tab2.layout.addWidget(self.tab2.label4,6,0)
        self.tab2.layout.addWidget(self.tab2.label8,6,1)
        self.tab2.layout.addWidget(self.tab2.ln4,7,1)
        self.tab2.layout.addWidget(self.tab2.ln8, 7,0)
        self.tab2.layout.addWidget(self.tab2.label9, 8,0) 
        self.tab2.layout.addWidget(self.tab2.ln9, 9,0)
        self.tab2.layout.addWidget(self.tab2.ln10, 11,0,2,2)
        self.tab2.layout.addWidget(self.tab2.pushButton1,10,0)
        self.tab2.layout.addWidget(self.tab2.pushButton2,10,1)
        dynamic_canvas = FigureCanvas(Figure(figsize=(9, 5)))
        self.tab2._dynamic_ax = dynamic_canvas.figure.subplots()
        self.tab2._dynamic_ax.tick_params(labelsize=6)
        self.tab2._dynamic_ax.grid()
        self.tab2._dynamic_ax.plot(x,y,'-o', color='black')
        self.tab2._timer = dynamic_canvas.new_timer(100, [(self._update_window, (), {})])
        self.tab2.pushButton1.clicked.connect(self.start_mm)
        self.tab2.pushButton2.clicked.connect(self.stop_mm)
        self.tab2.layout.addWidget(dynamic_canvas, 0, 3,15,4)
        self.tab2.setLayout(self.tab2.layout)

        '''
         # Create 3rd tab
        self.tab3.layout = QGridLayout(self.tab3)
        self.tab3.cb  = QComboBox(self)
        self.tab3.cb2 = QComboBox(self)
        self.tab3.cb3 = QComboBox(self)
        self.tab3.cb4 = QComboBox(self)
        self.tab3.ln1 = QLineEdit(self)
        self.tab3.ln1.setReadOnly(True)
        self.tab3.ln2 = QLineEdit(self)
        self.tab3.ln2.setReadOnly(True)
        self.tab3.ln3 = QLineEdit(self)
        self.tab3.ln3.setReadOnly(True)
        self.tab3.ln4 = QLineEdit(self)
        self.tab3.ln4.setReadOnly(True)
        self.tab3.label1 = QLabel("Solution Algorthim",self)
        self.tab3.cb.addItems(["Simulated Annealing", "Knapsack"])
        self.tab3.label2 = QLabel("Initial Solution",self)
        self.tab3.cb2.addItems(["Nearest Neighbour", "Sweeping"])
        self.tab3.label3 = QLabel("Scenario",self)
        self.tab3.cb3.addItems(["Moabit86", "Moabit220", "Mouint Jou Park"])
        self.tab3.label4 = QLabel("Number of MUERMELS",self)
        self.tab3.cb4.addItems(["1", "2", "3"])
        self.tab3.label5 = QLabel("Total Traveling Distance",self)
        self.tab3.label6 = QLabel("Solution Energy",self)
        self.tab3.label7 = QLabel("Solution Time",self)
        self.tab3.label8 = QLabel("Number of Dustbins",self)
        self.tab3.layout.addWidget(self.tab3.label1,0,0)
        self.tab3.layout.addWidget(self.tab3.label5,0,1)
        self.tab3.layout.addWidget(self.tab3.ln1,1,1)
        self.tab3.layout.addWidget(self.tab3.cb, 1,0)
        self.tab3.layout.addWidget(self.tab3.label2,2,0)
        self.tab3.layout.addWidget(self.tab3.label6,2,1)
        self.tab3.layout.addWidget(self.tab3.ln2,3,1)
        self.tab3.layout.addWidget(self.tab3.cb2, 3,0)
        self.tab3.layout.addWidget(self.tab3.label3,4,0)
        self.tab3.layout.addWidget(self.tab3.label7,4,1)
        self.tab3.layout.addWidget(self.tab3.ln3,5,1)
        self.tab3.layout.addWidget(self.tab3.cb3, 5,0)
        self.tab3.layout.addWidget(self.tab3.label4,6,0)
        self.tab3.layout.addWidget(self.tab3.label8,6,1)
        self.tab3.layout.addWidget(self.tab3.ln4,7,1)
        self.tab3.layout.addWidget(self.tab3.cb4, 7,0)
        self.tab3.setLayout(self.tab3.layout)
        '''
 
        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)
    
    

    @pyqtSlot()
    def on_click(self):
        print("HELOOOO")
        print(self.tab1.cb.currentText())
        print(self.tab1.cb2.currentText())
        print(self.tab1.cb3.currentText())
        print(self.tab1.cb4.currentText())
        self.tab1.ln1.setText("-------")
        self.tab1.ln5.setText("Sensor are working correclty........................................ ......... addWidget addWidget")
    def start_mm(self):
        print("HELOOOO")
        self.tab2._timer.start()
        self.tab2.ln10.setText("Sensor are working correclty........................................ ......... addWidget addWidget")
    def stop_mm(self):
        print("HELOOOO")
        self.tab2._timer.stop()
        self.tab2.ln10.setText("Sensor are working correclty........................................ ......... addWidget addWidget")

    def _update_window(self):
        self.tab2._dynamic_ax.clear()
        global x, y1, y2, y3, N, count_iter, last_number_clicks
        x.append(x[count_iter] + 0.01)
        y1.append(np.random.random())
        idx_inf = max([count_iter-N, 0])
        self.tab2._dynamic_ax.plot(x[idx_inf:count_iter], y1[idx_inf:count_iter],'-o', color='b')
        count_iter += 1
        self.tab2._dynamic_ax.figure.canvas.draw()
        
if __name__ == '__main__':
    #pressed_key = {}
    #clicks = []
    #last_number_clicks = len(clicks)
    N = 25
    y1 = [np.random.random()]
    x = [0]
    count_iter = 0
    #f_route = np.array(ks_trial.f_route)
    #x = f_route[:,1] #lon
    #y = f_route[:,0] #lat
    x = [13.34270598,13.34233706,13.33920884,13.33991503,13.33841079,13.3442691,13.35111111,13.35135288,13.35145139,13.35165766,13.35312095,13.33616661,13.3532823,13.35001144,13.35002178,13.35002642,13.34977776,13.34649924,13.3345659,13.34526884,13.34551357,13.34405863,13.34513639,13.3363824,13.33656147,13.33731384,13.33444444,13.33416667,13.33444444,13.33511097,13.33527778,13.33594557,13.33614326,13.33600072,13.33581217,13.33630281,13.33690426,13.33744376,13.33835885,13.34613734,13.34671365,13.34523743,13.34485096,13.34651478,13.35340122,13.33928379,13.33926014,13.338549,13.33807022,13.3373762,13.33668686,13.33613094,13.3343753,13.33479262,13.33384911,13.33433527,13.33472222,13.33519697,13.33987827,13.34045718,13.34088179,13.34170416,13.34216873,13.34279518,13.34145502,13.34145301,13.34145964,13.34147376,13.34145566,13.34121807,13.341461,13.34014664,13.33992987,13.33990075,13.34012036,13.33988681,13.34063178,13.34061383,13.33988282,13.34007645,13.33987386,13.34001341,13.33984945,13.33888889,13.34121078,13.34526557,13.34526122]
    y = [52.52841649,52.52818661,52.53212366,52.53160597,52.53284736,52.53241665, 52.52901199,52.52901619,52.52732931,52.52732018,52.52772173,52.53305018, 52.52699385,52.52913957,52.52699684,52.52769432,52.52913951,52.53328832, 52.53265722,52.52795176,52.52807604,52.53212215,52.53262458,52.53174174, 52.53243477,52.53296238,52.5313212 ,52.53101664,52.53039141,52.53055556, 52.53034364,52.52984154,52.529794,52.5290024 ,52.52902199,52.52860124, 52.52843635,52.52848975,52.52874938,52.53332643,52.53359008,52.53290376, 52.53369774,52.53553433,52.53668597,52.53028613,52.53003406,52.52959236, 52.53015929,52.53022358,52.5302711 ,52.53087487,52.53188674,52.53257988, 52.52964027,52.52879117,52.52855278,52.5286982 ,52.52847022,52.52819594, 52.52819357,52.52806806,52.52807619,52.52809571,52.52858405,52.52943333, 52.53013333,52.53069686,52.53111111,52.53105693,52.53183003,52.53188185, 52.53210656,52.53110167,52.53064745,52.53046789,52.52990327,52.53007269, 52.52968148,52.52907325,52.52906238,52.52731102,52.527287,52.53084744, 52.53326332,52.52910393,52.52690597]
    #print(ks_trial.f_route)
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())

