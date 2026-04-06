# 🧠 Autism Detection System

## 📌 Overview
This project is a deep learning-based Autism Detection System that classifies images into **Autistic** and **Non-Autistic** categories using transfer learning.

The model is built using **MobileNetV2** and optimized through fine-tuning and preprocessing techniques to improve prediction performance.

---

## 🚀 Results
- Achieved **~80% accuracy**
- Balanced precision and recall across both classes
- Improved performance using fine-tuning and learning rate optimization

---

## 🛠️ Tech Stack
- Python
- TensorFlow / Keras
- OpenCV
- NumPy, Matplotlib

---

## 🧠 Model Details
- Base Model: MobileNetV2 (Pre-trained on ImageNet)
- Technique: Transfer Learning + Fine-Tuning
- Optimizations:
  - Reduced learning rate
  - EarlyStopping
  - ReduceLROnPlateau
  - BatchNormalization & Dropout

---

## 📂 Project Structure
Autism/
├── main.py
├── README.md
├── requirements.txt
├── .gitignore


---

## 📊 Dataset
The dataset used for training is not included due to size limitations.

👉 You can access it here:  
(Add Google Drive / Kaggle link here)

---

## ▶️ How to Run

1. Clone the repository:
git clone https://github.com/divyankaaaaaa/autism-detection-system.git
2. Navigate to the project
cd autism-detection-system
3. Install dependencies:
pip install -r requirements.txt
4. Run the model:
python main.py

## 💡 Future Improvements
- Improve accuracy using EfficientNet
- Add real-time prediction system
- Deploy using Streamlit/Web App

---

## 👩‍💻 Author
Divyanka Tripathi
