import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QFileDialog, QLabel, QPushButton, QHBoxLayout, QMessageBox
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

        # Image label with initial text
        self.image_label = QLabel("Select an MRI image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFont(QFont("Arial", 22))

        # Button to select MRI image
        self.select_button = QPushButton("Select MRI Image")
        self.select_button.setFixedSize(200, 50)  # Set fixed size for the button
        self.select_button.setStyleSheet("""QPushButton {
            background-color:#876ca6;  color:white; padding:10px; border:none; border-radius:5px; font-size:16px;
            }
            QPushButton:hover {
                background-color: #e1cdf7;
            }""")  # Set font size
        self.select_button.clicked.connect(self.select_image)

        # Label to display prediction result
        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont("Arial", 14))

        # Layout for arranging elements
        layout = QVBoxLayout()
        layout.addWidget(self.image_label, alignment=Qt.AlignCenter)  # Align image label to center
        layout.addWidget(self.select_button, alignment=Qt.AlignCenter)  # Align button to center
        layout.addWidget(self.result_label, alignment=Qt.AlignCenter)  # Align result label to center
        
        # Copyright label
        copyright_label = QLabel("© Yassine Zirh & Ayoub Ennair. All rights reserved.")
        copyright_label.setAlignment(Qt.AlignCenter)
        copyright_label.setFont(QFont("Arial", 10))
        layout.addWidget(copyright_label)

        self.setLayout(layout)
        self.setWindowTitle("MRI Tumor Classification")
        self.setGeometry(100, 100, 400, 300)  # Set initial window size

        # Set background color
        self.setStyleSheet("background-color: lightgray;")  # Set a light gray background color

        self.show()

    def select_image(self):
        self.image_path, _ = QFileDialog.getOpenFileName(self, "Select MRI Image", "", "Image Files (*.jpg *.jpeg *.png)")
        if self.image_path:
            self.update_image()

    def update_image(self):
        try:
            # Load the selected image
            img = load_img(self.image_path, target_size=(240, 240))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Make prediction
            prediction = model.predict(img_array)
            class_prediction = np.round(prediction).astype(int)

            if class_prediction == 0:
                result_text = "The MRI image doesn't have a Tumor."
            else:
                result_text = "The MRI image has a tumor (Please consult a medical professional for diagnosis and treatment)."

            # Display selected image
            pixmap = QPixmap(self.image_path).scaled(240, 240, Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)

            # Update result label with prediction text
            self.result_label.setText(result_text)

        except Exception as e:
            print(f"Error loading image: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MRIApp()
    sys.exit(app.exec_())
