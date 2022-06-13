import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QWidget, QAction, QTabWidget,QVBoxLayout,QComboBox,QLabel,QGridLayout,QLineEdit
from PyQt5.QtGui import QIcon

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import Qt

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
        '''
        self.tabs.addTab(self.tab1,"Pre-Launch")
        self.tabs.addTab(self.tab2,"Active")
        self.tabs.addTab(self.tab3,"Post-Launch")        
        '''
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
