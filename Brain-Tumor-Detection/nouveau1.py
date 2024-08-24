import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QFileDialog, QLabel, QPushButton, QFrame
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.models import load_model

# Load the model
model = load_model('./besttransfermodel.keras')

class MRIApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.image_path = None

        # Create a frame to contain all the components
        self.frame = QFrame(self)
        self.frame.setGeometry(50, 50, 300, 200)  # Set the geometry of the frame
        self.frame.setStyleSheet("background-image: url('back.jpeg');")  # Set background image

        # Image label with initial text
        self.image_label = QLabel("Select an MRI image", parent=self.frame)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFont(QFont("Arial", 22))

        # Button to select MRI image
        self.select_button = QPushButton("Select MRI Image", parent=self.frame)
        self.select_button.setFixedSize(200, 50)
        self.select_button.setStyleSheet("""
            QPushButton {
                background-color: #876ca6;
                color: black;
                padding: 10px;
                border: none;
                border-radius: 5px;
                font-size: 20px;
            }
            QPushButton:hover {
                background-color: #e1cdf7;
            }
        """)
        self.select_button.clicked.connect(self.select_image)

        # Label to display prediction result
        self.result_label = QLabel("", parent=self.frame)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont("Arial", 14))

        # Layout for arranging elements inside the frame
        layout = QVBoxLayout(self.frame)
        layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
        layout.addWidget(self.select_button, alignment=Qt.AlignCenter)
        layout.addWidget(self.result_label, alignment=Qt.AlignCenter)
        self.frame.setLayout(layout)

        # Set the layout of the main window
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.frame)
        self.setLayout(main_layout)

        self.setWindowTitle("MRI Tumor Classification")
        self.setGeometry(100, 100, 400, 300)  # Set initial window size

        self.show()

    def select_image(self):
        self.image_path, _ = QFileDialog.getOpenFileName(self, "Select MRI Image", "", "Image Files (*.jpg *.jpeg *.png)")
        if self.image_path:
            self.update_image()

    def update_image(self):
        try:
            img = load_img(self.image_path, target_size=(240, 240))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            class_prediction = np.round(prediction).astype(int)

            if class_prediction == 0:
                result_text = "The MRI image doesn't have a Tumor."
            else:
                result_text = "The MRI image has a tumor (Please consult a medical professional for diagnosis and treatment)."

            pixmap = QPixmap(self.image_path).scaled(240, 240, Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)

            self.result_label.setText(result_text)

        except Exception as e:
            print(f"Error loading image: {e}")

def main():
    app = QApplication(sys.argv)
    window = MRIApp()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
