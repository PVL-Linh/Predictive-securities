import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton
from PyQt5.QtCore import QFile, QTextStream
from UI.sidebar_ui import Ui_MainWindow
from manager_data import CSVManager # type: ignore
from test import HomePageWidget  # Import HomePageWidget from test.py

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.icon_only_widget.hide()
        self.ui.stackedWidget.setCurrentIndex(0)
        self.ui.home_btn_2.setChecked(True)

        # Create the CSVManager instance and add it to the stacked widget
        self.csv_manager = CSVManager()
        self.ui.stackedWidget.addWidget(self.csv_manager)

        # Create the HomePageWidget instance and add it to the stacked widget
        self.home_page_widget = HomePageWidget()  # Initialize HomePageWidget
        self.ui.stackedWidget.addWidget(self.home_page_widget)

        # Connect the orders buttons to show the CSVManager
        self.ui.orders_btn_1.toggled.connect(self.show_csv_manager)
        self.ui.orders_btn_2.toggled.connect(self.show_csv_manager)
        
        # Connect the home buttons to show the HomePageWidget
        self.ui.home_btn_1.toggled.connect(self.show_home_page)
        self.ui.home_btn_2.toggled.connect(self.show_home_page)

        # Connect the dashboard buttons to show the HomePageWidget
        self.ui.dashborad_btn_1.toggled.connect(self.show_home_page)
        self.ui.dashborad_btn_2.toggled.connect(self.show_home_page)

    ## Function for searching
    def on_search_btn_clicked(self):
        self.ui.stackedWidget.setCurrentIndex(5)
        search_text = self.ui.search_input.text().strip()
        if search_text:
            self.ui.label_9.setText(search_text)

    ## Function for changing page to user page
    def on_user_btn_clicked(self):
        self.ui.stackedWidget.setCurrentIndex(6)

    ## Change QPushButton Checkable status when stackedWidget index changed
    def on_stackedWidget_currentChanged(self, index):
        btn_list = self.ui.icon_only_widget.findChildren(QPushButton) \
                    + self.ui.full_menu_widget.findChildren(QPushButton)
        
        for btn in btn_list:
            if index in [5, 6]:
                btn.setAutoExclusive(False)
                btn.setChecked(False)
            else:
                btn.setAutoExclusive(True)
            
    ## functions for changing menu page
    def on_home_btn_1_toggled(self):
        self.show_home_page()
    
    def on_home_btn_2_toggled(self):
        self.show_home_page()

    def on_dashborad_btn_1_toggled(self):
        self.show_home_page()

    def on_dashborad_btn_2_toggled(self):
        self.show_home_page()

    def on_orders_btn_1_toggled(self):
        self.show_csv_manager()

    def on_orders_btn_2_toggled(self):
        self.show_csv_manager()

    ## Function to show CSVManager
    def show_csv_manager(self):
        self.ui.stackedWidget.setCurrentIndex(self.ui.stackedWidget.indexOf(self.csv_manager))

    ## Function to show HomePageWidget
    def show_home_page(self):
        self.ui.stackedWidget.setCurrentIndex(self.ui.stackedWidget.indexOf(self.home_page_widget))

if __name__ == "__main__":
    app = QApplication(sys.argv)

    style_file = QFile("UI/style.qss")
    style_file.open(QFile.ReadOnly | QFile.Text)
    style_stream = QTextStream(style_file)
    app.setStyleSheet(style_stream.readAll())

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
