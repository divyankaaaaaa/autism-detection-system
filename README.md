# 🧠 Autism Detection System

A deep learning image classifier that distinguishes **Autistic** vs. **Non-Autistic**
facial images using transfer learning on MobileNetV2.

🔗 **Live demo:** `<add your Hugging Face Space link here once deployed>`

> ⚠️ **Disclaimer:** This is a research/educational project, **not a medical or
> diagnostic tool**. Autism is a clinical diagnosis made by qualified professionals
> through standardized behavioral assessment — not by classifying a single photo.
> This project exists to demonstrate a transfer-learning image classification
> pipeline, not to detect autism in practice.

---

## 📌 Overview

The model classifies facial images into two categories using **MobileNetV2**
(pre-trained on ImageNet) with a custom classification head, trained in two phases:
first the head alone, then fine-tuning the top layers of the base network at a
low learning rate.

## 🚀 Results

*(fill these in after you retrain with `src/train.py` — the script auto-generates
the plots referenced below)*

| Metric | Value |
|---|---|
| Test accuracy | `__%` |
| Precision | `__` |
| Recall | `__` |
| F1-score | `__` |

![Training curves](reports/training_curves.png)
![Confusion matrix](reports/confusion_matrix.png)

## 🛠️ Tech Stack

- Python, TensorFlow / Keras
- MobileNetV2 (transfer learning + fine-tuning)
- OpenCV, NumPy, Matplotlib, scikit-learn
- Streamlit (demo app)

## 🧠 Model Details

- **Base model:** MobileNetV2, pre-trained on ImageNet, `include_top=False`
- **Head:** GlobalAveragePooling → BatchNorm → Dropout → Dense(128, relu) → Dropout → Dense(1, sigmoid)
- **Training:** two-phase (frozen base → fine-tune top layers)
- **Regularization:** Dropout, BatchNormalization, EarlyStopping, ReduceLROnPlateau

## 📂 Project Structure

```
autism-detection-system/
├── app/
│   └── app.py              # Streamlit demo app
├── src/
│   ├── config.py            # paths & hyperparameters
│   ├── data_loader.py       # data generators + dataset splitting helper
│   ├── train.py              # two-phase training script
│   ├── evaluate.py           # test-set evaluation, confusion matrix
│   └── predict.py            # single-image CLI prediction
├── data/                     # dataset (not tracked in git — see data/README.md)
├── models/                   # trained model (.keras) saved here
├── reports/                  # training curves, confusion matrix, classification report
├── requirements.txt
└── README.md
```

## ▶️ How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/divyankaaaaaa/autism-detection-system.git
cd autism-detection-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your dataset (see data/README.md)

# 4. Train
python -m src.train

# 5. Evaluate on the test set
python -m src.evaluate

# 6. Predict on a single image
python -m src.predict path/to/image.jpg

# 7. Run the demo app locally
streamlit run app/app.py
```

## 🌐 Deployment (Hugging Face Spaces)

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space), SDK = **Streamlit**.
2. Push this repo's contents to the Space (or link the Space to this GitHub repo).
3. Make sure `models/autism_mobilenetv2.keras` is included in the Space (use Git LFS
   if the file is large — Hugging Face supports it natively).
4. Set the Space's app file to `app/app.py`.
5. The Space will build automatically from `requirements.txt`.

## 💡 Future Improvements

- Compare against EfficientNet / other backbones
- Add Grad-CAM visualizations so predictions are interpretable, not just a label
- Expand and diversify the training dataset
- Add automated tests and a CI pipeline (GitHub Actions)

## 👩‍💻 Author

Divyanka Tripathi
