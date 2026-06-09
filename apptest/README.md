# 🔬 CancerGrading Assistant

> **Breast Cancer Histopathology Image Classifier** — Automatically grades breast cancer tissue images into **Grade 1**, **Grade 2**, or **Grade 3**.

---

## 📋 Overview

This application uses deep learning models trained on breast cancer histopathology images to assist pathologists and doctors in cancer grading. Simply upload a tissue image, and the AI will predict the cancer grade with confidence scores.

| Grade | Meaning |
|-------|---------|
| **GRADE1** | Low grade (well-differentiated) — slower-growing, less aggressive |
| **GRADE2** | Intermediate grade (moderately-differentiated) |
| **GRADE3** | High grade (poorly-differentiated) — faster-growing, more aggressive |

---

## 🚀 Quick Start

### 1️⃣ Install Dependencies

Open a terminal/command prompt and run:

```bash
cd apptest
pip install -r requirements.txt
```

### 2️⃣ Launch the App

```bash
cd apptest
python app.py
```

Your browser will automatically open to **http://localhost:7860**

> **Note:** The application is fully self-contained. All model architectures and pre-trained checkpoint files are included inside `./apptest/`. No need to run training or reference external directories.

### 🏆 Alternative: Download Standalone EXE (No Python Required)

If the users don't have Python installed, you can build a **single-file Windows executable** that requires **zero dependencies**:

**Build the EXE (run once on a machine with Python):**
```bash
cd apptest
python build_exe.py
```

This will create: `dist/CancerGrading Assistant.exe` (~2-3 GB due to bundled PyTorch).

**To distribute:**
1. Copy `CancerGrading Assistant.exe` to any Windows computer
2. Copy the `checkpoints/` folder alongside it
3. Double-click the EXE — **no Python or pip needed!**

The app will open automatically at http://localhost:7860

---

## 🖥️ How to Use

### Tab 1: 📷 Single Image Classification

1. **Select a model** from the dropdown (Fusion models are recommended as they are most accurate)
2. **Upload** a histopathology image (JPG or PNG)
3. Click **"Classify"**
4. View:
   - **Predicted Grade** (large colored text)
   - **Confidence percentage**
   - **Confidence bars** showing probability distribution across all 3 grades

### Tab 2: 📁 Batch Processing

1. Upload **multiple images** at once
2. Click **"Process All"**
3. View the results table with each image's predicted grade and confidence
4. **Download** the results as a CSV/Excel-compatible report

### Tab 3: ⚖️ Model Comparison

1. Upload a **single image**
2. Click **"Compare All Models"**
3. See how each trained model performs on the same image
4. Useful for cross-validation or research purposes

---

## 🧠 Available Models

| Model | Description | Recommended For |
|-------|-------------|-----------------|
| **Fusion (Exp4)** | AlexNet + VGG16 + ResNet50 ensemble with GELU + SE Attention | ✅ **Best overall accuracy** |
| **VGG19 Fusion** | VGG19 + AlexNet + ResNet50 ensemble with GELU + SE Attention | ✅ Excellent accuracy |
| **VGG19+GELU+SE** | VGG19 with GELU activation and SE Attention | High accuracy |
| **VGG19+GELU** | VGG19 with GELU activation only | Good accuracy |
| **Exp3 Models** | AlexNet/VGG16/ResNet50 + GELU + SE Attention | Good accuracy |
| **Exp2 Models** | AlexNet/VGG16/ResNet50 + GELU only | Moderate accuracy |
| **Exp1 Models** | AlexNet/VGG16/ResNet50 + ReLU (baseline) | Baseline comparison |

> 💡 **Tip:** The Fusion model combines multiple neural networks and generally provides the most reliable predictions. Start with Fusion, then cross-check with individual models if needed.

---

## 🌐 Deployment Options

### Local Use (Single Computer)
```bash
cd apptest
python app.py
```
Access at: http://localhost:7860

### Hospital Network (LAN)
The app already binds to `0.0.0.0`, so other computers on your network can access it at:
```
http://YOUR-IP:7860
```

### Public Access (via Gradio sharing)
Edit `app.py` and change `share=False` to `share=True`, then run:
```bash
cd apptest
python app.py
```
A public URL (e.g., `https://xxxxx.gradio.live`) will be generated.

---

## 📁 File Structure

```
apptest/                          # Fully self-contained application
├── app.py                        # Main Gradio web interface
├── predict.py                    # Prediction engine
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── models/                       # Model architectures (copied locally)
│   ├── alexnet.py
│   ├── vgg16.py
│   ├── vgg19.py
│   ├── resnet50.py
│   ├── fusion.py
│   ├── attention.py
│   └── __init__.py
│
└── checkpoints/                  # Pre-trained model weights (.pth)
      ├── Exp4_Fusion.pth
      ├── ExpVGG19_Fusion_GELU_SE.pth
      ├── ExpVGG19_VGG19_GELU_SE.pth
      ├── ExpVGG19_VGG19_GELU.pth
      ├── ExpVGG19_VGG19_ReLU.pth
      ├── Exp3_AlexNet.pth
      ├── Exp3_VGG16.pth
      ├── Exp3_ResNet50.pth
      ├── Exp2_AlexNet.pth
      ├── Exp2_VGG16.pth
      ├── Exp2_ResNet50.pth
      ├── Exp1_AlexNet.pth
      ├── Exp1_VGG16.pth
      └── Exp1_ResNet50.pth
```

---

## ⚠️ Important Notes

- **This is a clinical decision support tool**, not a diagnostic device. Always verify results with expert pathological review.
- The app works best with images that match the training data format (histopathology tissue sections).
- For best performance, use images with similar resolution and staining as the training dataset.
- Model loading may take a few seconds on first use (checkpoints are cached after initial load).

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| App won't start | Run `pip install -r requirements.txt` to ensure all dependencies are installed |
| "No module named 'models'" | Make sure you're running from inside the `apptest/` directory |
| Out of memory | Select a lighter model (Exp1/Exp2) instead of Fusion |
| Slow prediction | First prediction loads the model from disk; subsequent predictions are cached |

---

## 📞 Support

For technical issues or questions, please contact the development team or open an issue in the project repository.

---

*Built with PyTorch & Gradio — © CancerGrading Project*