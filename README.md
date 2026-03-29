---
title: VisionSpeak IVIS
emoji: 👁️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---

# 👁️ VisionSpeak: IVIS
## Intelligent Vision Interpretation System

| Field | Detail |
|---|---|
| **Student** | Avasarala Sai Ganesh |
| **Roll No** | 24M11MC007 |
| **Programme** | MCA |
| **Department** | Computer Applications |
| **Institution** | Aditya University |
| **Academic Year** | 2025–2026 |
| **Project Guide** | Dr. Bapuji Rao |
| **IEEE Reference** | 10968127 |

---

## 📁 Project Files

```
VisionSpeak/
├── app.py                    ← Main Streamlit app (upload here first)
├── requirements.txt          ← Python dependencies
├── packages.txt              ← System deps (Tesseract OCR)
├── VisionSpeak_Colab.ipynb   ← Google Colab notebook (10 steps)
└── README.md                 ← This file
```

---

## 📌 Abstract

Visually impaired individuals face difficulties accessing printed text and Braille information. **VisionSpeak: IVIS** is an assistive system that converts both printed text and Braille into audible speech using computer vision and AI. It uses OpenCV for preprocessing, Tesseract 5 for OCR, a CNN for Braille recognition, and gTTS for speech output.

**Keywords:** Assistive Technology · Braille Recognition · OCR · Computer Vision · TTS · Deep Learning

---

## 🚀 OPTION A — Deploy on Hugging Face Spaces (FREE + Permanent)

### Step 1: Create Account
Go to https://huggingface.co → Sign Up → Verify email

### Step 2: Create New Space
1. https://huggingface.co/new-space
2. **Space name:** `VisionSpeak`
3. **SDK:** `Streamlit`
4. **Hardware:** `CPU Basic` (Free)
5. Click **Create Space**

### Step 3: Upload Files
In your Space → Files tab → Add file → Upload files → Upload all 4:
- `app.py`
- `requirements.txt`
- `packages.txt`
- `README.md`

Click **Commit changes to main**

### Step 4: Wait for Build (~3–5 mins)
Watch the **App** tab for logs. Once green → your app is live at:
```
https://YOUR_USERNAME-VisionSpeak.hf.space
```

### Step 5: Test
- **Tab 1:** Upload any printed text image → Extract & Speak
- **Tab 2:** Upload a Braille image → Decode Braille & Speak
- **Tab 3:** Use camera to capture text in real time
- Download the MP3 audio output

---

## 🧪 OPTION B — Google Colab (with CNN Training)

### Step 1: Open Notebook
1. https://colab.research.google.com
2. File → Upload notebook → Upload `VisionSpeak_Colab.ipynb`

### Step 2: Enable GPU
Runtime → Change runtime type → **T4 GPU** → Save

### Step 3: Get ngrok Token
1. https://ngrok.com → Sign up free
2. Dashboard → Your Authtoken → Copy
3. Paste in Step 10: `NGROK_TOKEN = 'your_token'`

### Step 4: Run All Cells in Order

| Step | What It Does |
|------|-------------|
| 1 | Install Tesseract (system) |
| 2 | Install Python packages |
| 3 | Import & verify all libraries |
| 4 | OpenCV preprocessing pipeline test |
| 5 | Tesseract OCR test |
| 6 | CNN model definition + training (20 epochs on synthetic data) |
| 7 | Braille recognition test |
| 8 | Text-to-Speech audio test |
| 9 | Accuracy metrics charts (CAR, WAR, CNN accuracy) |
| 10 | Launch app via ngrok (public URL) |

### Step 5: Get Public URL
After Step 10, you'll see:
```
🚀 VisionSpeak: IVIS is LIVE at: https://xxxx.ngrok-free.app
```

---

## 💻 OPTION C — Run Locally

```bash
# 1. Install Tesseract
# Ubuntu:  sudo apt install tesseract-ocr tesseract-ocr-eng
# macOS:   brew install tesseract
# Windows: https://github.com/UB-Mannheim/tesseract/wiki

# 2. Setup Python environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Run
streamlit run app.py
# Opens: http://localhost:8501
```

---

## 🧠 System Architecture

```
Input Image (Upload / Camera)
        │
        ▼
OpenCV Preprocessing
  ├─ Grayscale conversion
  ├─ Fast Non-Local Means Denoising
  ├─ CLAHE contrast normalisation
  ├─ Adaptive Gaussian Thresholding
  └─ Morphological open/close
        │
   ┌────┴────┐
   ▼         ▼
Tesseract   CNN Braille
LSTM OCR    Classifier
(PSM-6)     (3-block, 27 classes)
   │         │
   └────┬────┘
        ▼
   Text Output
        │
        ▼
  gTTS Engine
  (English/Hindi/Kannada/Tamil/Telugu)
        │
        ▼
   MP3 Audio Output
```

---

## 📊 Performance Targets

| Metric | Target |
|---|---|
| OCR Character Accuracy (CAR) | ≥ 92% |
| OCR Word Accuracy (WAR) | ≥ 88% |
| Braille Recognition (single char) | ≥ 85% |
| Braille Recognition (sentence) | ≥ 78% |
| TTS Latency | < 3 s |
| End-to-End Latency | < 5 s |

---

## 🔧 Troubleshooting

**Tesseract not found:**
```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Linux/HF
```

**No Braille dots detected:**
- Use 300+ DPI images
- Ensure black dots on white background
- Avoid shadows, use flat scan

**gTTS network error in Colab:**
```bash
!pip install --upgrade gtts
```

**HF Spaces build fails:**
- `packages.txt` must have `tesseract-ocr` (no version number)
- Check logs in the App tab

---

**VisionSpeak: IVIS · MCA Final Project · Aditya University · 2025–2026**
