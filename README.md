# 🩻 AI Radiologist: Pneumonia Detection
"Detecting Pneumonia with the precision of Deep Learning and the transparency of Explainable AI."

## 🩺 The Problem
Pneumonia remains the single largest infectious cause of death in children worldwide, claiming the lives of over 700,000 children under 5 years old annually (UNICEF/WHO). Early diagnosis is critical, but expert radiologists aren't always available in remote areas. Misdiagnosis (false negatives) can be fatal.

## 💡 The Solution
**AI Radiologist** is a deep learning diagnostic tool that:
- Analyzes chest X-rays in milliseconds.
- Detects signs of pneumonia with 91% accuracy.
- Explains its decision using Grad-CAM, highlighting the exact infected regions in the lungs so doctors can trust the result.

## 🔬 How It Works
**1. The Brain (VGG16 Transfer Learning)**

We didn't start from scratch. We used VGG16, a powerful Convolutional Neural Network (CNN) pre-trained on millions of images.
- We **fine-tuned** the final layers to recognize the subtle "cloudy" patterns (opacities) typical of pneumonia.
- Optimization: We used Class Weighting to handle imbalances and a low learning rate (1e-5) to carefully adjust the model's weights.

**2. The "Why" (Grad-CAM Visualization)**

Black-box AI is dangerous in medicine. We implemented **Gradient-weighted Class Activation Mapping (Grad-CAM)**.

- This technique tracks the gradients flowing into the final convolutional layer.
- It generates a heatmap (Red/Yellow) showing exactly where the model is "looking."
- If the model predicts Pneumonia, the heatmap lights up the infected lobes of the lungs.

## 📊 Performance Metrics

<img width="710" height="205" alt="Screenshot 2025-12-02 at 8 37 42 PM" src="https://github.com/user-attachments/assets/0a9e7fdf-2655-4f8b-ac5e-4e915d22546b" />

**Note:** In medical AI, Recall is King. We prioritized Recall to ensure no sick patient is sent home mistakenly.

## 🛠️ Installation & Local Setup

Want to run this on your own machine? Follow these steps.

1. **Clone the Repository**
   ```bash
    git clone https://github.com/kg7825881/AI-Radiologist
    ```
2. Install Dependencies
   ```bash
    pip install -r requirements.txt
    ```
4. Download the Model
This project uses Git LFS for the heavy model file.
   ```bash
   git lfs pull
    ```

6. Run the App
   ```bash
    streamlit run app.py
    ```

## 📂 Project Structure
  ```bash
     ai-radiologist/
     ├── app.py                      # Main Streamlit Application
     ├── requirements.txt            # Python Dependencies
     ├── pneumonia_detection_model_vgg16.keras  # The Trained AI Brain (100MB+)
     └── README.md                   # Project Documentation
  ```

## ⚠️ Disclaimer
This tool is for educational purposes only. It is not a certified medical device and should not be used for real-world diagnosis without clinical validation. Always consult a qualified doctor.

## 👨‍💻 Credits

**Dataset:** Chest X-Ray Images (Pneumonia) from Kaggle.
