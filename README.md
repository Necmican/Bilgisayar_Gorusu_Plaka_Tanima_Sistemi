# 🚗 Automatic License Plate Recognition System (ALPR)

A real-time **License Plate Recognition** system developed with Python, utilizing **YOLOv8** for vehicle detection and **EasyOCR** for optical character recognition.

This project can process both **images** and **video streams** to detect license plates, read the text, and automatically log the data into a CSV file with timestamps.

## 🚀 Features

* **Dual Mode:** Seamlessly switch between **Image** and **Video** processing modes via a terminal menu.
* **High Performance:** Implements **frame skipping** and **image upscaling** techniques to optimize OCR accuracy and processing speed in videos.
* **Automatic Logging:** Detected license plates are automatically saved to a `plaka_kayitlari.csv` file with date and time information.
* **Smart Filtering:** Filters out non-plate text and small objects to reduce false positives.
* **Auto-Resize:** Automatically resizes high-resolution inputs to fit the screen without losing aspect ratio.

## 🛠️ Tech Stack

* **[Python 3.x](https://www.python.org/)** - Core programming language.
* **[YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics)** - State-of-the-art object detection model.
* **[EasyOCR](https://github.com/JaidedAI/EasyOCR)** - AI-based Optical Character Recognition.
* **[OpenCV](https://opencv.org/)** - Real-time computer vision and image processing.

## 📦 Installation

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/Plaka-Tanima-Sistemi.git](https://github.com/YOUR_USERNAME/Plaka-Tanima-Sistemi.git)
    cd Plaka-Tanima-Sistemi
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

    *Note: If you have an NVIDIA GPU, make sure to install the CUDA-supported version of PyTorch for better performance.*

3.  **Run the Application**
    ```bash
    python main.py
    ```

## 🎮 Usage

When you run `main.py`, a menu will appear:

1.  **Image Mode:** Enter the path of an image file (e.g., `.jpg`, `.png`). The system will detect and display the license plate.
2.  **Video Mode:** Enter the path of a video file (e.g., `.mp4`). The system will track cars and read plates in real-time.
3.  **Exit:** Closes the application.

## 📂 Project Structure
