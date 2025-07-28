import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QLabel, QComboBox, QTableWidget, QTableWidgetItem, QHeaderView, QProgressDialog
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from model_du_doan import train_model_lstm, train_model_GRU, train_model_RNN
import os
from hienThi import HienThi
from duDoan import truc_quan_bieu_do_cot,truc_quan_bieu_do_mien

class TrainModelThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)

    def __init__(self, model_type, parent=None):
        super(TrainModelThread, self).__init__(parent)
        self.model_type = model_type

    def run(self):
        if self.model_type == "LSTM":
            train_model_lstm()
        elif self.model_type == "SVR":
            train_model_GRU()
        elif self.model_type == "XGBoost":
            train_model_RNN()
        self.finished.emit(self.model_type)

class HomePageWidget(QWidget): 
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Dự đoán chứng khoán')
        self.setGeometry(100, 100, 1000, 800)
        
        self.data = None

        main_layout = QVBoxLayout()

        # Tiêu đề hiển thị tên file
        self.fileLabel = QLabel('No file loaded')
        self.fileLabel.setStyleSheet("font-size: 18px; font-weight: bold;")
        main_layout.addWidget(self.fileLabel)

        # Nút bấm
        button_layout = QHBoxLayout()
        
        self.loadButton = QPushButton('Nạp')
        self.loadButton.setStyleSheet("""
            QPushButton {
                background-color: lightblue; 
                font-size: 16px; 
                border-radius: 10px;
                border: 1px solid #5c5c5c;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #add8e6;
            }
        """)
        self.loadButton.clicked.connect(self.load_data)
        button_layout.addWidget(self.loadButton)

        self.clearButton = QPushButton('Xóa')
        self.clearButton.setStyleSheet("""
            QPushButton {
                background-color: lightcoral; 
                font-size: 16px; 
                border-radius: 10px;
                border: 1px solid #5c5c5c;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #f08080;
            }
        """)
        self.clearButton.clicked.connect(self.clear_data)
        button_layout.addWidget(self.clearButton)

        self.trainButton = QPushButton('TRAIN')
        self.trainButton.setStyleSheet("""
            QPushButton {
                background-color: lightgreen; 
                font-size: 16px; 
                border-radius: 10px;
                border: 1px solid #5c5c5c;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #90ee90;
            }
        """)
        self.trainButton.clicked.connect(self.train_model)
        button_layout.addWidget(self.trainButton)

        self.modelComboBox = QComboBox()
        self.modelComboBox.setStyleSheet("""
            QComboBox {
                font-size: 16px; 
                border-radius: 10px;
                border: 1px solid #5c5c5c;
                padding: 5px 10px;
            }
        """)
        self.modelComboBox.addItems(["LSTM", "SVR", "XGBoost"])
        button_layout.addWidget(self.modelComboBox)

        main_layout.addLayout(button_layout)

        # ComboBox chọn cột và loại biểu đồ
        combo_layout = QHBoxLayout()

        column_label = QLabel('Chọn cột để trực quan hóa:')
        column_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        combo_layout.addWidget(column_label)

        self.columnComboBox = QComboBox()
        self.columnComboBox.setStyleSheet("""
            QComboBox {
                font-size: 16px; 
                border-radius: 10px;
                border: 1px solid #5c5c5c;
                padding: 5px 10px;
            }
        """)
        self.columnComboBox.currentIndexChanged.connect(self.visualize_data)
        combo_layout.addWidget(self.columnComboBox)

        chart_type_label = QLabel('Chọn loại biểu đồ:')
        chart_type_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        combo_layout.addWidget(chart_type_label)

        self.chartTypeComboBox = QComboBox()
        self.chartTypeComboBox.setStyleSheet("""
            QComboBox {
                font-size: 16px; 
                border-radius: 10px;
                border: 1px solid #5c5c5c;
                padding: 5px 10px;
            }
        """)
        self.chartTypeComboBox.addItems(["Bar", "Line"])
        self.chartTypeComboBox.currentIndexChanged.connect(self.visualize_data)
        combo_layout.addWidget(self.chartTypeComboBox)

        # ComboBox chọn khoảng thời gian
        time_period_label = QLabel('Month & Year')
        time_period_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        combo_layout.addWidget(time_period_label)

        self.timePeriodComboBox = QComboBox()
        self.timePeriodComboBox.setStyleSheet("""
            QComboBox {
                font-size: 16px; 
                border-radius: 10px;
                border: 1px solid #5c5c5c;
                padding: 5px 10px;
            }
        """)
        self.timePeriodComboBox.addItems(["Month", "Year"])
        self.timePeriodComboBox.currentIndexChanged.connect(self.visualize_data)
        combo_layout.addWidget(self.timePeriodComboBox)
        main_layout.addLayout(combo_layout)

        # Bảng hiển thị dữ liệu
        table_layout = QVBoxLayout()
        table_label = QLabel('Dữ liệu:')
        table_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        table_layout.addWidget(table_label)

        self.tableWidget = QTableWidget()
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableWidget.setStyleSheet("""
            QTableWidget {
                border: 1px solid #5c5c5c;
                border-radius: 10px;
                font-size: 14px;
                padding: 5px;
            }
            QTableWidget::item {
                border: none;
                padding: 5px;
            }
        """)
        table_layout.addWidget(self.tableWidget)
        main_layout.addLayout(table_layout)

        # Biểu đồ trực quan hóa
        plot_layout = QVBoxLayout()
        plot_label = QLabel('Biểu đồ trực quan hóa:')
        plot_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        plot_layout.addWidget(plot_label)

        self.canvas = FigureCanvas(plt.figure())
        plot_layout.addWidget(self.canvas)
        main_layout.addLayout(plot_layout)

        # Nút hiển thị
        display_button_layout = QHBoxLayout()

        prediction_type_label = QLabel('Chọn loại dự đoán:')
        prediction_type_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        display_button_layout.addWidget(prediction_type_label)

        self.predictionTypeComboBox = QComboBox()
        self.predictionTypeComboBox.setStyleSheet("""
            QComboBox {
                font-size: 16px; 
                border-radius: 10px;
                border: 1px solid #5c5c5c;
                padding: 5px 10px;
            }
        """)
        self.predictionTypeComboBox.addItems(["Column", "Range"])
        display_button_layout.addWidget(self.predictionTypeComboBox)
        
        self.displayButton = QPushButton('Hiển thị')
        self.displayButton.setStyleSheet("""
            QPushButton {
                background-color: lightgrey; 
                font-size: 16px; 
                border-radius: 10px;
                border: 1px solid #5c5c5c;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #d3d3d3;
            }
        """)
        self.displayButton.clicked.connect(self.hienthi)
        display_button_layout.addWidget(self.displayButton)

        self.displayPredictionButton = QPushButton('Hiển thị dự đoán')
        self.displayPredictionButton.setStyleSheet("""
            QPushButton {
                background-color: lightgrey; 
                font-size: 16px; 
                border-radius: 10px;
                border: 1px solid #5c5c5c;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #d3d3d3;
            }
        """)
        self.displayPredictionButton.clicked.connect(self.dudoan)
        display_button_layout.addWidget(self.displayPredictionButton)

        main_layout.addLayout(display_button_layout)

        self.setLayout(main_layout)

    def load_data(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        fileName, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if fileName:
            self.data = pd.read_csv(fileName)
            self.save_data_to_csv(fileName)
            self.display_data()
            self.populate_columns()
            self.visualize_data()
            self.fileLabel.setText(f'Loaded file: {fileName}')

    def save_data_to_csv(self, fileName):
        save_path = 'data1/data.csv'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if self.data is not None:
            self.data.to_csv(save_path, index=False)
            print(f'Data saved to {save_path}')

    def display_data(self):
        if self.data is not None:
            self.tableWidget.setRowCount(self.data.shape[0])
            self.tableWidget.setColumnCount(self.data.shape[1])
            self.tableWidget.setHorizontalHeaderLabels(self.data.columns)

            for i in range(self.data.shape[0]):
                for j in range(self.data.shape[1]):
                    self.tableWidget.setItem(i, j, QTableWidgetItem(str(self.data.iat[i, j])))

    def populate_columns(self):
        if self.data is not None:
            self.columnComboBox.clear()
            numeric_columns = [col for col in self.data.columns if pd.api.types.is_numeric_dtype(self.data[col]) and col not in ['data', 'company_name']]
            self.columnComboBox.addItems(numeric_columns[:12])  # Chỉ thêm 12 cột đầu tiên

    def visualize_data(self):
        if self.data is not None and self.columnComboBox.count() > 0:
            selected_column = self.columnComboBox.currentText()
            chart_type = self.chartTypeComboBox.currentText()
            time_period = self.timePeriodComboBox.currentText()  # Get the time period selection
            
            if selected_column and pd.api.types.is_numeric_dtype(self.data[selected_column]):
                self.canvas.figure.clear()
                ax = self.canvas.figure.add_subplot(111)
                
                # Ensure 'Date' column is parsed as datetime
                if 'Date' in self.data.columns:
                    self.data['Date'] = pd.to_datetime(self.data['Date'])
                    if time_period == "Month":
                        self.data['MonthYear'] = self.data['Date'].dt.to_period('M')  # Extract month and year
                        data_to_plot = self.data.groupby('MonthYear')[selected_column].mean().iloc[:]  # Limit to 70 data points
                    elif time_period == "Year":
                        self.data['Year'] = self.data['Date'].dt.year  # Extract year
                        data_to_plot = self.data.groupby('Year')[selected_column].mean().iloc[:]  # Limit to 70 data points

                if chart_type == "Bar":
                    data_to_plot.plot(kind='bar', ax=ax)
                elif chart_type == "Line":
                    data_to_plot.plot(kind='line', ax=ax)

                ax.set_xlabel(time_period)  # Set the x-axis label to 'Month/Year' or 'Year'
                ax.set_ylabel(selected_column)  # Set the y-axis label to the selected column
                self.canvas.draw()
            else:
                self.canvas.figure.clear()
                ax = self.canvas.figure.add_subplot(111)
                ax.text(0.5, 0.5, 'No numeric data to plot', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                self.canvas.draw()

    def clear_data(self):
        self.data = None
        self.tableWidget.clear()
        self.columnComboBox.clear()
        self.canvas.figure.clear()
        self.canvas.draw()
        self.fileLabel.setText('No file loaded')

    def train_model(self):
        selected_model = self.modelComboBox.currentText()
        self.progressDialog = QProgressDialog("Training model...", "Cancel", 10, 100, self)
        self.progressDialog.setWindowTitle("Please Wait")
        self.progressDialog.setWindowModality(Qt.WindowModal)
        self.progressDialog.setValue(0)
        self.progressDialog.setStyleSheet("""
            QProgressDialog {
                background-color: #f0f0f0;
                border: 2px solid #5c5c5c;
                border-radius: 10px;
                padding: 10px;
            }
            QLabel {
                font-size: 14px;
            }
            QProgressBar {
                border: 1px solid #5c5c5c;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: lightgreen;
                width: 20px;
            }
        """)
        self.progressDialog.show()

        self.trainThread = TrainModelThread(selected_model)
        self.trainThread.progress.connect(self.progressDialog.setValue)
        self.trainThread.finished.connect(self.on_train_finished)
        self.trainThread.start()

    def on_train_finished(self, model_type):
        self.progressDialog.setValue(100)
        self.progressDialog.close()
        print(f"Training of {model_type} model finished.")

    def hienthi(self):
        HienThi()
        print("Chức năng hiển thị được gọi")
        # Thêm mã cho chức năng hiển thị tại đây

    def dudoan(self):
        prediction_type = self.predictionTypeComboBox.currentText()
        if prediction_type == "Column":
            self.predict_by_column()
        elif prediction_type == "Range":
            self.predict_by_range()
        
    def predict_by_column(self):
        truc_quan_bieu_do_cot()
        print("Dự đoán dựa trên cột")
        # Thêm mã cho dự đoán dựa trên cột tại đây

    def predict_by_range(self):
        truc_quan_bieu_do_mien()
        print("Dự đoán dựa trên miền")
        # Thêm mã cho dự đoán dựa trên miền tại đây

    def saiso(self):
        print("Chức năng sai số được gọi")
        # Thêm mã cho chức năng sai số tại đây

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = HomePageWidget()  # Create an instance of HomePageWidget
    window.show()
    sys.exit(app.exec_())
