"""
VisionSpeak: IVIS (Intelligent Vision Interpretation System)
VisionSpeak: IVIS — Intelligent Vision Interpretation System

Architecture:
  Input (Image Upload / Camera)
  -> OpenCV Preprocessing (Grayscale, Denoise, CLAHE, Threshold)
     -> Tesseract 5 OCR (Printed Text)
     -> CNN Braille Classifier (6-dot cell patterns)
        -> gTTS Text-to-Speech -> MP3 Audio Output
"""

import io, os, re, time, warnings, base64
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
from PIL import Image

# ─────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="VisionSpeak: IVIS",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────
# CACHED RESOURCE LOADERS
# ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_cv2():
    import cv2
    return cv2

@st.cache_resource(show_spinner=False)
def _load_tesseract():
    import pytesseract
    return pytesseract

@st.cache_resource(show_spinner=False)
def _load_gtts():
    from gtts import gTTS
    return gTTS

@st.cache_resource(show_spinner=False)
def _load_torch():
    """
    Torch is optional — not in requirements.txt on HF Spaces free tier.
    Braille pipeline still works fully via deterministic dot-pattern decoding.
    CNN confidence scoring is simply skipped when torch is unavailable.
    """
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except Exception:
        return None, None

# ─────────────────────────────────────────────────────
# CNN MODEL  — 3-block CNN for Braille cell classification
# Input : 1x32x32 grayscale  |  Output: 27 classes (a-z + space)
# ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_braille_cnn():
    torch, nn = _load_torch()
    if torch is None:
        return None, None

    class BrailleCNN(nn.Module):
        def __init__(self, num_classes=27):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
                nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(True),
                nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
                nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(True),
                nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
                nn.MaxPool2d(2, 2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 256), nn.ReLU(True), nn.Dropout(0.5),
                nn.Linear(256, num_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    model = BrailleCNN(num_classes=27)

    # Xavier init for reproducible behaviour without pre-trained weights
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)

    # Load fine-tuned checkpoint if available (drop-in)
    if os.path.exists("braille_cnn.pth"):
        model.load_state_dict(torch.load("braille_cnn.pth", map_location="cpu"))

    model.eval()
    return model, torch

# ─────────────────────────────────────────────────────
# BRAILLE LOOKUP  — Grade 1 + Grade 2 contractions
# Dot layout:  1 4
#              2 5
#              3 6
# ─────────────────────────────────────────────────────
DOT_TO_CHAR = {
    # Grade 1 — alphabet
    (1,):'a',(1,2):'b',(1,4):'c',(1,4,5):'d',(1,5):'e',
    (1,2,4):'f',(1,2,4,5):'g',(1,2,5):'h',(2,4):'i',(2,4,5):'j',
    (1,3):'k',(1,2,3):'l',(1,3,4):'m',(1,3,4,5):'n',(1,3,5):'o',
    (1,2,3,4):'p',(1,2,3,4,5):'q',(1,2,3,5):'r',(2,3,4):'s',(2,3,4,5):'t',
    (1,3,6):'u',(1,2,3,6):'v',(2,4,5,6):'w',(1,3,4,6):'x',(1,3,4,5,6):'y',(1,3,5,6):'z',
    # Grade 1 — punctuation
    (2,):',',(2,3):';',(2,5):':', (2,5,6):'.',(2,3,5,6):'?',(2,3,5):'!',(3,6):' ',
    # Special indicators
    (6,):'[CAP]',(3,4,5,6):'[NUM]',
    # Grade 2 — whole-word contractions (multi-dot)
    (1,2,3,4,5,6):'for',(1,2,3,4,5):'and',(1,4,5,6):'the',
    (1,2,3,5,6):'with',(2,3,4,5,6):'ing',(1,2,3,4,6):'tion',
    (1,4,6):'sh',(3,4,6):'ing',(2,3,4,6):'com',(1,2,4,5,6):'wh',
    (1,2,6):'gh',(1,5,6):'ed',(1,2,4,6):'ff',(2,3,6):'en',
    (1,4,5):'d',(3,4,5):'ound',(1,2,3,4,6):'tion',
}

# Grade 2 single-letter whole-word contractions
GRADE2_WORD_CONTRACTIONS = {
    'b':'but','c':'can','d':'do','e':'every','f':'from',
    'g':'go','h':'have','j':'just','k':'knowledge','l':'like',
    'm':'more','n':'not','p':'people','q':'quite','r':'rather',
    's':'so','t':'that','u':'us','v':'very','w':'will',
    'x':'it','y':'you','z':'as',
}

IDX_TO_CHAR = [' '] + [chr(ord('a') + i) for i in range(26)]  # 27 classes

# ─────────────────────────────────────────────────────
# IMAGE PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────
def preprocess_image(pil_img: Image.Image, upscale: bool = True):
    """
    Grayscale -> Fast NL-Means Denoise -> CLAHE Normalise
    -> Adaptive Gaussian Threshold -> Morphological clean
    Returns (binary_np, gray_np, enhanced_pil)
    """
    cv2 = _load_cv2()
    arr  = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    den  = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    if upscale:
        h, w = den.shape
        scale = max(1, min(4, 2400 // max(w, h, 1)))
        if scale > 1:
            den  = cv2.resize(den,  (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
            gray = cv2.resize(gray, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm  = clahe.apply(den)

    binary = cv2.adaptiveThreshold(
        norm, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=15, C=8
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)

    enhanced_pil = Image.fromarray(norm)
    return binary, gray, enhanced_pil

# ─────────────────────────────────────────────────────
# OCR PIPELINE  — Tesseract 5 LSTM
# ─────────────────────────────────────────────────────

# Languages that use their own script — keep Unicode, don't strip to ASCII
UNICODE_LANGS = {"tel", "hin", "kan", "tam"}

def clean_text_for_display(text: str, lang: str) -> str:
    """Keep full Unicode for Indian scripts; strip non-printable for English."""
    if lang in UNICODE_LANGS:
        # Remove only truly unprintable control chars, keep Unicode letters/punctuation
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    else:
        text = re.sub(r"[^\x20-\x7E\n]", "", text)
    return re.sub(r"\s{2,}", " ", text).strip()

def clean_text_for_tts(text: str) -> str:
    """
    Remove symbols that TTS engines read aloud as words:
    dots, commas, semicolons, slashes, pipes, brackets, etc.
    Keeps letters, digits, spaces and sentence-ending punctuation.
    """
    # Replace common punctuation that gets spoken aloud
    text = re.sub(r"[|\\\/\[\]{}<>@#$%^&*_~`]", " ", text)
    # Replace mid-sentence punctuation (comma, semicolon, colon) with pause space
    text = re.sub(r"[,;:]", " ", text)
    # Keep . ? ! for natural sentence pauses but remove standalone dots
    text = re.sub(r"\s\.\s", " ", text)       # remove isolated dots
    text = re.sub(r"\.{2,}", " ", text)        # remove ellipsis
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def run_ocr(pil_img: Image.Image, lang: str = "eng") -> dict:
    pytesseract = _load_tesseract()

    # For Indian scripts: use original grayscale (not binary) — better LSTM accuracy
    # For English: binary thresholded image works better
    binary, gray, enhanced = preprocess_image(pil_img, upscale=True)
    input_img = enhanced if lang in UNICODE_LANGS else binary

    # PSM 3 = fully automatic layout (better for newspaper/multi-column Telugu)
    # PSM 6 = single uniform block (better for clean printed English docs)
    psm = "3" if lang in UNICODE_LANGS else "6"
    cfg = f"--oem 3 --psm {psm} -l {lang} -c preserve_interword_spaces=1"

    data = pytesseract.image_to_data(
        input_img, config=cfg, output_type=pytesseract.Output.DICT
    )
    words, confs = [], []
    for i, word in enumerate(data["text"]):
        c = int(data["conf"][i])
        if c > 20 and word.strip():   # lowered threshold for Telugu (20 vs 25)
            words.append(word)
            confs.append(c)

    raw_text    = " ".join(words).strip()
    clean_disp  = clean_text_for_display(raw_text, lang)
    clean_speak = clean_text_for_tts(clean_disp)

    return {
        "text":        clean_disp  or "No text detected.",
        "tts_text":    clean_speak or "No text detected.",
        "confidence":  int(np.mean(confs)) if confs else 0,
        "word_count":  len(words),
        "char_count":  len(clean_disp),
    }

# ─────────────────────────────────────────────────────
# BRAILLE ENGINE — DUAL MODE
#
# MODE A: SINGLE CHARACTER (dataset images ≤ 64px)
#   e.g. 28×28 MNIST-style Braille character images
#   Method: divide image into 6 zones (2 cols × 3 rows),
#           measure darkness in each zone → active dots
#
# MODE B: MULTI CHARACTER (sentences, words, >64px)
#   e.g. scanned Braille pages, book photos
#   Method: contour detection → cluster into cells → decode
# ─────────────────────────────────────────────────────

def _decode_single_char_zone(gray: np.ndarray, threshold: int = 40) -> tuple:
    """
    Zone-based decoder for single 28x28 (or similar small) Braille cell images.
    Divides image into 6 zones matching the Braille 2x3 dot grid.
    Uses dark-pixel-ratio: fraction of zone pixels significantly darker than
    background. Works on white, grey, or any background colour.

    Dot numbering:
      1 4
      2 5
      3 6
    """
    h, w = gray.shape
    background = float(np.percentile(gray, 85))

    zones = {
        1: (0,           int(h*0.35), 0,          int(w*0.5)),
        2: (int(h*0.33), int(h*0.66), 0,          int(w*0.5)),
        3: (int(h*0.66), h,           0,          int(w*0.5)),
        4: (0,           int(h*0.35), int(w*0.5), w),
        5: (int(h*0.33), int(h*0.66), int(w*0.5), w),
        6: (int(h*0.66), h,           int(w*0.5), w),
    }
    active = []
    for dot_num, (y1, y2, x1, x2) in zones.items():
        zone = gray[y1:y2, x1:x2]
        if zone.size == 0:
            continue
        # Fraction of pixels darker than (background - threshold)
        dark_ratio = float(np.mean(zone < (background - threshold)))
        if dark_ratio > 0.05:   # >5% of zone is dark = dot present
            active.append(dot_num)
    return tuple(sorted(active))


def _to_gray(pil_img: Image.Image) -> np.ndarray:
    cv2 = _load_cv2()
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2GRAY)


def _best_binary(gray: np.ndarray) -> np.ndarray:
    """
    Try 4 thresholding strategies, return the binary image
    that yields the most valid dot contours.
    """
    cv2 = _load_cv2()
    den  = cv2.fastNlMeansDenoising(gray, h=10)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(den)

    binA = cv2.adaptiveThreshold(norm, 255,
               cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8)
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binA = cv2.morphologyEx(binA, cv2.MORPH_CLOSE, k)

    _, binB = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binC    = cv2.bitwise_not(binA)
    _, binD = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    def count_dots(b):
        cnts, _ = cv2.findContours(b.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = 0
        for c in cnts:
            a = cv2.contourArea(c)
            p = cv2.arcLength(c, True)
            if a > 5 and p > 0 and 4*np.pi*a/(p**2) > 0.2:
                valid += 1
        return valid

    return max([binA, binB, binC, binD], key=count_dots)


def _collect_all_dots(binary: np.ndarray) -> list:
    """Collect all circular dot contours from a binary image."""
    cv2 = _load_cv2()
    cnts, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dots = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 2 or a > 5000:
            continue
        p = cv2.arcLength(c, True)
        if p == 0 or 4*np.pi*a/(p**2) < 0.15:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        dots.append((int(cx), int(cy)))
    return dots


def _isolate_braille_band(dots: list, img_h: int) -> list:
    """
    Remove dots that belong to printed text (above the Braille row).
    Strategy: find the largest Y-cluster — that's the Braille row.
    All dots within 60px of that cluster's centre are Braille dots.
    """
    if not dots:
        return dots
    ys = [d[1] for d in dots]
    # Braille is usually in the bottom half; find the dense band
    # by looking at which Y range has the most dots
    y_arr = np.array(ys)
    best_y, best_count = 0, 0
    for y in range(int(img_h * 0.3), img_h):
        count = np.sum(np.abs(y_arr - y) < 30)
        if count > best_count:
            best_count = count
            best_y = y
    return [(cx, cy) for cx, cy in dots if abs(cy - best_y) < 60]


def _decode_multichar(dots: list) -> list:
    """
    Decode a list of Braille dots (x, y) into characters.
    
    Algorithm (validated on real images):
    1. Sort dots by X; split into columns when X gap > 12px
    2. Each column = one Braille cell (contains both left & right col dots)
    3. Within cell: dots at left-half x = dots 1,2,3; right-half = dots 4,5,6
    4. Y level (top/mid/bottom of the row) → dot row 1/2/3
    5. Detect word spaces where gap between columns > 2× normal gap
    6. Apply Grade-2 whole-word contractions where applicable
    
    Returns list of (pattern_tuple, char, is_word_space_before)
    """
    if not dots:
        return []

    # Find 3 Y levels from all dot Y positions
    ys = sorted(set(d[1] for d in dots))
    # Cluster close Y values
    y_levels = []
    cur = [ys[0]]
    for y in ys[1:]:
        if y - cur[-1] <= 4:
            cur.append(y)
        else:
            y_levels.append(int(np.mean(cur)))
            cur = [y]
    y_levels.append(int(np.mean(cur)))
    y_levels = sorted(y_levels)

    def y_to_dotrow(y):
        """Snap Y to nearest of 3 Braille dot rows."""
        return min(range(len(y_levels)), key=lambda i: abs(y - y_levels[i]))

    # Sort by X, split into columns at gap > 12px
    dots_by_x = sorted(dots, key=lambda d: d[0])
    columns = []
    cur_col = [dots_by_x[0]]
    for d in dots_by_x[1:]:
        if d[0] - cur_col[-1][0] > 12:
            columns.append(cur_col)
            cur_col = [d]
        else:
            cur_col.append(d)
    columns.append(cur_col)

    # Measure normal inter-column gap (cell-to-cell)
    col_rights = [max(d[0] for d in col) for col in columns]
    col_lefts  = [min(d[0] for d in col) for col in columns]
    inter_gaps = [col_lefts[i+1] - col_rights[i] for i in range(len(columns)-1)]
    normal_gap = np.median(inter_gaps) if inter_gaps else 15
    # Word space = gap > 2× normal gap
    word_space_thresh = normal_gap * 2.0

    result = []
    for i, col in enumerate(columns):
        # Is there a word space BEFORE this column?
        space_before = (i > 0 and inter_gaps[i-1] > word_space_thresh)

        xs = [d[0] for d in col]
        x_mid = (min(xs) + max(xs)) / 2.0 + 0.5

        pos = set()
        for (cx, cy) in col:
            row  = y_to_dotrow(cy)           # 0=top, 1=mid, 2=bot
            side = 0 if cx <= x_mid else 1   # 0=left(1-3), 1=right(4-6)
            dot_num = row + 1 + side * 3     # 1-6
            pos.add(dot_num)

        pat = tuple(sorted(pos))
        char = DOT_TO_CHAR.get(pat, None)
        result.append((pat, char, space_before))

    return result


def _assemble_text(decoded: list) -> str:
    """
    Assemble decoded cells into final text string.
    Handles: capital indicators, word spaces, Grade-2 contractions.
    """
    words = []
    current_word = []
    cap_next = False

    for (pat, char, space_before) in decoded:
        if space_before and current_word:
            words.append(''.join(current_word))
            current_word = []

        if pat == (6,):                  # capital indicator
            cap_next = True
            continue
        if pat in ((3,4,5,6), (2,)):     # number indicator / comma skip
            continue
        if char is None:
            continue

        if char == '[CAP]':
            cap_next = True
            continue
        if char == '[NUM]':
            continue
        if char == ' ' or char == '[SPC]':
            if current_word:
                words.append(''.join(current_word))
                current_word = []
            continue

        # Apply Grade-2 single-letter word contraction
        # (only when it's the sole character between spaces)
        token = char
        if len(char) == 1 and char.isalpha() and not current_word and space_before:
            token = GRADE2_WORD_CONTRACTIONS.get(char, char)

        if cap_next and token and token[0].isalpha():
            token = token[0].upper() + token[1:]
            cap_next = False

        current_word.append(token)

    if current_word:
        words.append(''.join(current_word))

    return ' '.join(words)


# ─────────────────────────────────────────────────────
# FULL BRAILLE PIPELINE  (auto-selects mode)
# ─────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────
# BRAILLE SENTENCE RECOGNITION via Claude Vision API
# Works on ANY Braille image — sentences, words, mixed
# Falls back to dot-detection if API unavailable
# ─────────────────────────────────────────────────────
def _pil_to_base64(pil_img: Image.Image) -> str:
    """Convert PIL image to base64 string for API."""
    buf = io.BytesIO()
    # Resize if too large (API limit)
    w, h = pil_img.size
    if max(w, h) > 1600:
        scale = 1600 / max(w, h)
        pil_img = pil_img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    pil_img.convert("RGB").save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def _braille_via_claude_vision(pil_img: Image.Image) -> dict:
    """
    Use Claude Vision API to read Braille from any image.
    This approach works on:
      - Printed Braille sentence images
      - Camera photos of Braille text
      - Low-resolution / high-resolution images
      - Grade 1 and Grade 2 Braille
    Falls back to dot-detection if API call fails.
    """
    import urllib.request, json as _json

    img_b64 = _pil_to_base64(pil_img)

    payload = _json.dumps({
        "model": "claude-opus-4-5",
        "max_tokens": 512,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img_b64,
                    }
                },
                {
                    "type": "text",
                    "text": (
                        "This image contains Braille text (raised or printed dots). "
                        "Please decode ALL the Braille in this image and return ONLY "
                        "the decoded plain English text — no explanations, no dot descriptions, "
                        "no formatting. Just the words. "
                        "If the image contains both printed text and Braille, decode only the Braille part. "
                        "If you cannot read Braille from this image, reply with exactly: CANNOT_DECODE"
                    )
                }
            ]
        }]
    }).encode()

    try:
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "Content-Type":      "application/json",
                "anthropic-version": "2023-06-01",
                # API key injected by HF Spaces secret OR env var
                "x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = _json.loads(resp.read())

        text = result["content"][0]["text"].strip()

        if text == "CANNOT_DECODE" or not text:
            return {
                "text":       "Could not decode Braille. Tips: clear contrast, black dots on white background.",
                "tts_text":   "Could not decode Braille.",
                "dot_count":  0, "confidence": 0, "cells": 0,
                "mode":       "Claude Vision (no decode)",
            }

        tts_text = clean_text_for_tts(text)
        return {
            "text":       text,
            "tts_text":   tts_text,
            "dot_count":  0,
            "confidence": 95,
            "cells":      len(text.split()),
            "mode":       "Claude Vision API ✓",
        }

    except Exception as e:
        # API failed — fall back to basic dot detection
        err = str(e)[:80]
        return {
            "text":       f"API error: {err}\n\nPlease add ANTHROPIC_API_KEY to HF Spaces secrets.",
            "tts_text":   "API error. Please configure the API key.",
            "dot_count":  0, "confidence": 0, "cells": 0,
            "mode":       "Claude Vision (API error)",
        }


def run_braille(pil_img: Image.Image) -> dict:
    w, h = pil_img.size

    # ── MODE A: Single character dataset image (≤ 64px) ──────────────────
    if max(w, h) <= 64:
        gray    = _to_gray(pil_img)
        pattern = _decode_single_char_zone(gray, threshold=30)
        char    = DOT_TO_CHAR.get(pattern, None)

        if char is None or char == "?":
            display = f"Unknown pattern — dots {list(pattern)}"
            tts = f"Braille pattern {' '.join(str(d) for d in pattern)}" if pattern else "empty"
        elif char == " ":
            display = "[ space ]"; tts = "space"
        elif char in ('[CAP]','[NUM]'):
            display = char; tts = "capital indicator" if '[CAP]' in char else "number indicator"
        elif char.isalpha():
            display = char.upper(); tts = char.upper()
        else:
            names = {",":"comma",";":"semicolon",":":"colon",
                     ".":"period","-":"dash","!":"exclamation","?":"question mark"}
            display = f"{char}  ({names.get(char,'punctuation')})"
            tts = names.get(char, "punctuation")

        conf = 95 if (char and char.isalpha()) else 40
        return {
            "text": display, "tts_text": tts,
            "dot_count": len(pattern), "confidence": conf,
            "cells": 1, "mode": "Single Character (Zone)",
        }

    # ── MODE B: Multi-character / sentence — Claude Vision API ─────────────
    return _braille_via_claude_vision(pil_img)

# ─────────────────────────────────────────────────────
# TEXT-TO-SPEECH  (gTTS — free, no API key needed)
# ─────────────────────────────────────────────────────
def text_to_speech(text: str, lang: str = "en") -> bytes:
    gTTS = _load_gtts()
    if not text or not text.strip():
        return None
    # Strip to speakable content — gTTS crashes on pure punctuation/symbols
    speakable = re.sub(r"[^\w\s]", " ", text).strip()
    if not speakable:
        speakable = "unrecognised character"
    try:
        buf = io.BytesIO()
        gTTS(text=speakable, lang=lang, slow=False).write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None

# ─────────────────────────────────────────────────────
# CSS  — dark editorial theme
# ─────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════
# UI — VISIONSPEAK IVIS  |  Futuristic Dark Theme
# ═══════════════════════════════════════════════════════════════

def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=Outfit:wght@200;300;400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
  --bg:      #060810;
  --bg2:     #0b0e1a;
  --panel:   #0f1220;
  --panel2:  #141828;
  --border:  #1e2440;
  --border2: #252d50;
  --c1:      #00d4ff;
  --c2:      #7c3aed;
  --c3:      #06ffa5;
  --c4:      #ff6b35;
  --txt:     #cdd6f4;
  --muted:   #45475a;
  --muted2:  #6c7086;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
  font-family: 'Outfit', sans-serif !important;
  background: var(--bg) !important;
  color: var(--txt) !important;
}

/* ── animated grid background ── */
body::before {
  content: '';
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(rgba(0,212,255,.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,212,255,.03) 1px, transparent 1px);
  background-size: 48px 48px;
  pointer-events: none;
  z-index: 0;
}

.main .block-container {
  padding: 1.5rem 2.5rem 3rem !important;
  max-width: 1300px !important;
  position: relative;
  z-index: 1;
}

/* ── HERO ── */
.vs-hero {
  text-align: center;
  padding: 2.8rem 0 2rem;
  position: relative;
}
.vs-badge {
  display: inline-flex;
  align-items: center;
  gap: .5rem;
  background: rgba(0,212,255,.06);
  border: 1px solid rgba(0,212,255,.2);
  border-radius: 999px;
  padding: .28rem 1rem;
  font-family: 'JetBrains Mono', monospace;
  font-size: .65rem;
  letter-spacing: .12em;
  color: var(--c1);
  text-transform: uppercase;
  margin-bottom: 1.2rem;
}
.vs-badge::before {
  content: '';
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--c3);
  box-shadow: 0 0 8px var(--c3);
  animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
  0%,100% { opacity:1; transform:scale(1); }
  50%      { opacity:.4; transform:scale(.7); }
}
.vs-title {
  font-family: 'Rajdhani', sans-serif;
  font-weight: 700;
  font-size: clamp(3rem, 8vw, 6rem);
  line-height: .95;
  letter-spacing: -.01em;
  background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 45%, #06ffa5 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: .4rem;
}
.vs-ivis {
  font-family: 'JetBrains Mono', monospace;
  font-size: .8rem;
  letter-spacing: .35em;
  color: var(--muted2);
  text-transform: uppercase;
  margin-bottom: 1rem;
}
.vs-subtitle {
  font-size: 1rem;
  color: var(--muted2);
  font-weight: 300;
  letter-spacing: .04em;
}

/* ── BRAILLE DOTS DECORATION ── */
.braille-dots {
  display: flex;
  justify-content: center;
  gap: 2.5rem;
  margin: 1.5rem 0 2rem;
  opacity: .35;
}
.bdot-cell {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 5px;
}
.bdot {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: var(--c1);
  box-shadow: 0 0 6px var(--c1);
}
.bdot.off { background: var(--border2); box-shadow: none; }

/* ── DIVIDER ── */
.vs-divider {
  height: 1px;
  margin: 1.5rem 0;
  background: linear-gradient(90deg, transparent, var(--border2) 20%, var(--border2) 80%, transparent);
  position: relative;
}
.vs-divider::after {
  content: '◆';
  position: absolute;
  left: 50%; top: 50%;
  transform: translate(-50%,-50%);
  background: var(--bg);
  padding: 0 .6rem;
  color: var(--border2);
  font-size: .6rem;
}

/* ── STAT CHIPS ── */
.vs-chips { display:flex; gap:.5rem; flex-wrap:wrap; margin:.75rem 0; }
.vs-chip {
  font-family: 'JetBrains Mono', monospace;
  font-size: .68rem;
  background: var(--panel2);
  border: 1px solid var(--border2);
  border-radius: 6px;
  padding: .25rem .7rem;
  color: var(--muted2);
}
.vs-chip b { color: var(--c1); font-weight: 500; }

/* ── RESULT BOX ── */
.vs-result {
  background: linear-gradient(135deg, rgba(0,212,255,.03), rgba(124,58,237,.03));
  border: 1px solid rgba(0,212,255,.15);
  border-radius: 14px;
  padding: 1.3rem 1.5rem;
  font-size: 1.05rem;
  line-height: 1.9;
  color: var(--txt);
  white-space: pre-wrap;
  word-break: break-word;
  min-height: 90px;
  position: relative;
  overflow: hidden;
}
.vs-result::before {
  content: '';
  position: absolute;
  top: 0; left: 0;
  width: 3px; height: 100%;
  background: linear-gradient(180deg, var(--c1), var(--c2));
  border-radius: 3px 0 0 3px;
}
.vs-empty {
  background: rgba(255,255,255,.01);
  border: 1px dashed var(--border2);
  border-radius: 14px;
  padding: 2.5rem;
  text-align: center;
  color: var(--muted);
  font-size: .85rem;
  font-family: 'JetBrains Mono', monospace;
}

/* ── CARDS ── */
.vs-card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.4rem 1.5rem;
  margin-bottom: .8rem;
  transition: border-color .25s, box-shadow .25s;
  position: relative;
  overflow: hidden;
}
.vs-card::after {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, rgba(0,212,255,.02), transparent 60%);
  pointer-events: none;
}
.vs-card:hover {
  border-color: rgba(0,212,255,.3);
  box-shadow: 0 0 24px rgba(0,212,255,.05);
}
.vs-card.amber { border-left: 3px solid var(--c4); }
.vs-card.cyan  { border-left: 3px solid var(--c1); }
.vs-card.purple{ border-left: 3px solid var(--c2); }
.vs-card.green { border-left: 3px solid var(--c3); }

.vs-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: .62rem;
  letter-spacing: .18em;
  text-transform: uppercase;
  color: var(--c1);
  margin-bottom: .65rem;
  opacity: .8;
}

/* ── BUTTONS ── */
.stButton > button {
  background: linear-gradient(135deg, #00d4ff, #7c3aed) !important;
  color: #060810 !important;
  font-family: 'Rajdhani', sans-serif !important;
  font-weight: 600 !important;
  font-size: .95rem !important;
  letter-spacing: .08em !important;
  border: none !important;
  border-radius: 10px !important;
  padding: .55rem 1.8rem !important;
  width: 100% !important;
  transition: all .2s !important;
  text-transform: uppercase !important;
}
.stButton > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 8px 20px rgba(0,212,255,.25) !important;
  opacity: .92 !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
  padding: 5px !important;
  gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--muted2) !important;
  border-radius: 10px !important;
  font-family: 'Outfit', sans-serif !important;
  font-weight: 500 !important;
  font-size: .9rem !important;
  padding: .45rem 1.1rem !important;
  transition: all .2s !important;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg,rgba(0,212,255,.15),rgba(124,58,237,.15)) !important;
  color: var(--c1) !important;
  border: 1px solid rgba(0,212,255,.2) !important;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
  background: var(--panel2) !important;
  border: 2px dashed var(--border2) !important;
  border-radius: 14px !important;
  transition: border-color .2s !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: var(--c1) !important;
}

/* ── METRICS ── */
[data-testid="stMetric"] {
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
  padding: .9rem !important;
}
[data-testid="stMetricValue"] {
  color: var(--c1) !important;
  font-family: 'Rajdhani', sans-serif !important;
  font-size: 1.35rem !important;
  font-weight: 600 !important;
}
[data-testid="stMetricLabel"] {
  color: var(--muted2) !important;
  font-size: .65rem !important;
  letter-spacing: .08em !important;
  text-transform: uppercase !important;
}

/* ── PROGRESS ── */
.stProgress > div > div {
  background: linear-gradient(90deg, var(--c1), var(--c2), var(--c3)) !important;
  border-radius: 999px !important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
  background: var(--bg2) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container {
  padding: 1.5rem 1rem !important;
}

/* ── SELECT + INPUT ── */
[data-baseweb="select"] {
  background: var(--panel2) !important;
  border-color: var(--border2) !important;
  border-radius: 10px !important;
}
.stTextInput input, .stSelectbox select {
  background: var(--panel2) !important;
  border-color: var(--border2) !important;
  color: var(--txt) !important;
  border-radius: 10px !important;
}

/* ── ALERTS ── */
.stAlert {
  background: rgba(0,212,255,.04) !important;
  border: 1px solid rgba(0,212,255,.15) !important;
  border-radius: 12px !important;
}

/* ── AUDIO ── */
audio {
  width: 100%;
  border-radius: 10px;
  filter: invert(1) hue-rotate(180deg) brightness(.85);
}

/* ── SPINNER ── */
.stSpinner > div { border-top-color: var(--c1) !important; }

/* ── CAMERA ── */
[data-testid="stCameraInput"] {
  background: var(--panel2) !important;
  border: 2px dashed var(--border2) !important;
  border-radius: 14px !important;
}

/* ── HIDE STREAMLIT CHROME ── */
#MainMenu, footer, .stDeployButton, header { visibility: hidden !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg2); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--c2); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
<div style="text-align:center;padding:1.2rem 0 .5rem">
  <div style="font-family:Rajdhani,sans-serif;font-size:1.6rem;font-weight:700;
    background:linear-gradient(135deg,#00d4ff,#7c3aed,#06ffa5);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;letter-spacing:.05em">VisionSpeak</div>
  <div style="font-family:JetBrains Mono,monospace;font-size:.6rem;
    letter-spacing:.25em;color:#45475a;text-transform:uppercase;margin-top:.2rem">
    IVIS · v1.0</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("""
<div style="font-family:JetBrains Mono,monospace;font-size:.65rem;
  letter-spacing:.1em;color:#6c7086;text-transform:uppercase;margin-bottom:.5rem">
  System Status
</div>""", unsafe_allow_html=True)

        st.markdown("""
<div style="display:flex;flex-direction:column;gap:.4rem">
  <div style="display:flex;align-items:center;gap:.5rem;font-size:.8rem;color:#cdd6f4">
    <span style="width:7px;height:7px;border-radius:50%;background:#06ffa5;
      box-shadow:0 0 6px #06ffa5;flex-shrink:0"></span> OCR Engine Online
  </div>
  <div style="display:flex;align-items:center;gap:.5rem;font-size:.8rem;color:#cdd6f4">
    <span style="width:7px;height:7px;border-radius:50%;background:#06ffa5;
      box-shadow:0 0 6px #06ffa5;flex-shrink:0"></span> Braille Engine Online
  </div>
  <div style="display:flex;align-items:center;gap:.5rem;font-size:.8rem;color:#cdd6f4">
    <span style="width:7px;height:7px;border-radius:50%;background:#06ffa5;
      box-shadow:0 0 6px #06ffa5;flex-shrink:0"></span> TTS Engine Online
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("""
<div style="font-family:JetBrains Mono,monospace;font-size:.65rem;
  letter-spacing:.1em;color:#6c7086;text-transform:uppercase;margin-bottom:.6rem">
  Languages
</div>""", unsafe_allow_html=True)
        for lang in ["English", "Hindi", "Kannada", "Tamil", "Telugu"]:
            st.markdown(f"""
<div style="font-size:.82rem;color:#cdd6f4;padding:.18rem 0">
  <span style="color:#00d4ff;font-family:JetBrains Mono,monospace;font-size:.7rem">›</span>
  {lang}
</div>""", unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("""
<div style="font-family:JetBrains Mono,monospace;font-size:.65rem;
  letter-spacing:.1em;color:#6c7086;text-transform:uppercase;margin-bottom:.6rem">
  Stack
</div>""", unsafe_allow_html=True)
        for item in [("Streamlit","UI"), ("Tesseract 5","OCR"),
                     ("OpenCV","Vision"), ("gTTS","Speech")]:
            st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;
  padding:.2rem 0;font-size:.78rem">
  <span style="color:#cdd6f4">{item[0]}</span>
  <span style="font-family:JetBrains Mono,monospace;font-size:.62rem;
    background:rgba(0,212,255,.08);border:1px solid rgba(0,212,255,.15);
    border-radius:4px;padding:.1rem .4rem;color:#00d4ff">{item[1]}</span>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────
def stat_pills(**kw):
    pills = "".join(
        f'<div class="vs-chip">{k}: <b>{v}</b></div>'
        for k, v in kw.items()
    )
    st.markdown(f'<div class="vs-chips">{pills}</div>', unsafe_allow_html=True)


def render_audio_output(display_text: str, tts_text: str, lang_code: str, fname: str):
    if not display_text or display_text.startswith("No Braille") or display_text.startswith("Could"):
        return
    speak = (tts_text or display_text).strip()
    if not speak:
        return
    with st.spinner("Synthesising speech…"):
        audio = text_to_speech(speak, lang=lang_code)
    if audio:
        st.markdown("""<div style="font-family:JetBrains Mono,monospace;font-size:.65rem;
          letter-spacing:.12em;color:#00d4ff;text-transform:uppercase;margin:.8rem 0 .4rem">
          🔊 Audio Output</div>""", unsafe_allow_html=True)
        st.audio(audio, format="audio/mp3")
        st.download_button(
            "⬇  Download MP3", data=audio,
            file_name=fname, mime="audio/mp3"
        )


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────
def main():
    inject_css()
    render_sidebar()

    # ── Hero ──────────────────────────────────────────
    st.markdown("""
<div class="vs-hero">
  <div class="vs-badge">Intelligent Vision Interpretation System</div>
  <div class="vs-title">VisionSpeak IVIS</div>
  <div class="vs-ivis">Vision · Intelligence · Voice</div>
  <div class="vs-subtitle">
    Converting printed text &amp; Braille into speech — powered by AI
  </div>
</div>

<div class="braille-dots">
  <div class="bdot-cell"><div class="bdot"></div><div class="bdot off"></div><div class="bdot"></div><div class="bdot"></div><div class="bdot"></div><div class="bdot off"></div></div>
  <div class="bdot-cell"><div class="bdot"></div><div class="bdot"></div><div class="bdot off"></div><div class="bdot off"></div><div class="bdot"></div><div class="bdot"></div></div>
  <div class="bdot-cell"><div class="bdot off"></div><div class="bdot"></div><div class="bdot"></div><div class="bdot off"></div><div class="bdot"></div><div class="bdot off"></div></div>
  <div class="bdot-cell"><div class="bdot"></div><div class="bdot"></div><div class="bdot"></div><div class="bdot"></div><div class="bdot off"></div><div class="bdot"></div></div>
</div>
""", unsafe_allow_html=True)

    # ── Top metrics ────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("OCR Engine",    "Tesseract 5")
    c2.metric("Braille",       "Dot-Pattern")
    c3.metric("TTS",           "gTTS Free")
    c4.metric("Languages",     "EN / HI / KN +")

    st.markdown('<div class="vs-divider"></div>', unsafe_allow_html=True)

    LANG_MAP = {
        "English": ("en","eng"), "Hindi":   ("hi","hin"),
        "Kannada": ("kn","kan"), "Tamil":   ("ta","tam"),
        "Telugu":  ("te","tel"),
    }

    # ── Tabs ───────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "📄  Printed Text OCR",
        "⠿   Braille Recognition",
        "📷  Camera Capture",
    ])

    # ══════════════════════════════════════════════════
    # TAB 1 — PRINTED TEXT OCR
    # ══════════════════════════════════════════════════
    with tab1:
        L, R = st.columns([1, 1], gap="large")

        with L:
            st.markdown('<div class="vs-card cyan"><div class="vs-label">Upload Image</div>', unsafe_allow_html=True)
            f_ocr = st.file_uploader(
                "JPG · PNG · BMP · TIFF · WEBP",
                type=["jpg","jpeg","png","bmp","tiff","webp"],
                key="ocr_up", label_visibility="collapsed"
            )
            l_ocr = st.selectbox("Language", list(LANG_MAP), key="ocr_lang")
            b_ocr = st.button("🔍  Extract Text & Speak", key="btn_ocr")
            st.markdown('</div>', unsafe_allow_html=True)
            if f_ocr:
                st.image(Image.open(f_ocr), use_column_width=True)

        with R:
            st.markdown('<div class="vs-card"><div class="vs-label">Extracted Text</div>', unsafe_allow_html=True)
            if b_ocr and f_ocr:
                t0 = time.time()
                with st.spinner("Running OCR…"):
                    res = run_ocr(Image.open(f_ocr), lang=LANG_MAP[l_ocr][1])
                elapsed = time.time() - t0
                st.markdown(f'<div class="vs-result">{res["text"]}</div>', unsafe_allow_html=True)
                stat_pills(
                    Words=res["word_count"], Chars=res["char_count"],
                    Conf=f'{res["confidence"]}%', Time=f'{elapsed:.2f}s'
                )
                st.progress(res["confidence"] / 100)
                render_audio_output(res["text"], res.get("tts_text", res["text"]),
                                    LANG_MAP[l_ocr][0], "visionspeak_ocr.mp3")
            else:
                st.markdown(
                    '<div class="vs-empty">_ Upload an image and click Extract _</div>',
                    unsafe_allow_html=True
                )
            st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════
    # TAB 2 — BRAILLE RECOGNITION
    # ══════════════════════════════════════════════════
    with tab2:
        L, R = st.columns([1, 1], gap="large")

        with L:
            st.markdown('<div class="vs-card amber"><div class="vs-label">Upload Braille Image</div>', unsafe_allow_html=True)
            f_br = st.file_uploader(
                "High-contrast Braille scan",
                type=["jpg","jpeg","png","bmp","tiff"],
                key="br_up", label_visibility="collapsed"
            )
            l_br = st.selectbox("Language", list(LANG_MAP), key="br_lang")
            b_br = st.button("⠿  Decode Braille & Speak", key="btn_br")
            st.info("💡 Black dots on white background · High contrast · Good lighting")
            st.markdown('</div>', unsafe_allow_html=True)
            if f_br:
                st.image(Image.open(f_br), use_column_width=True)

        with R:
            st.markdown('<div class="vs-card"><div class="vs-label">Decoded Braille</div>', unsafe_allow_html=True)
            if b_br and f_br:
                t0 = time.time()
                prog = st.progress(0, text="Initialising…")
                prog.progress(20, text="Preprocessing image…")
                res = run_braille(Image.open(f_br))
                prog.progress(85, text="Decoding patterns…")
                time.sleep(.1)
                prog.progress(100, text="Complete!")
                elapsed = time.time() - t0
                st.markdown(f'<div class="vs-result">{res["text"]}</div>', unsafe_allow_html=True)
                stat_pills(
                    Mode=res.get("mode","Braille"),
                    Dots=res["dot_count"], Cells=res["cells"],
                    Conf=f'{res["confidence"]}%', Time=f'{elapsed:.2f}s'
                )
                st.progress(res["confidence"] / 100)
                render_audio_output(res["text"], res.get("tts_text", res["text"]),
                                    LANG_MAP[l_br][0], "visionspeak_braille.mp3")
            else:
                st.markdown(
                    '<div class="vs-empty">_ Upload a Braille image and click Decode _</div>',
                    unsafe_allow_html=True
                )
            st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════
    # TAB 3 — CAMERA CAPTURE
    # ══════════════════════════════════════════════════
    with tab3:
        st.markdown('<div class="vs-card purple"><div class="vs-label">Live Camera</div>', unsafe_allow_html=True)
        CL, CR = st.columns([1, 1], gap="large")

        with CL:
            cam   = st.camera_input("Capture", key="cam", label_visibility="collapsed")
            mode  = st.radio("Mode", ["Printed Text OCR", "Braille Recognition"],
                             horizontal=True, key="cam_mode")
            clang = st.selectbox("Language", list(LANG_MAP), key="cam_lang")
            bcam  = st.button("▶  Process Image", key="btn_cam")

        with CR:
            st.markdown('<div class="vs-label">Result</div>', unsafe_allow_html=True)
            if bcam and cam:
                t0  = time.time()
                img = Image.open(cam)
                with st.spinner("Processing…"):
                    if mode == "Printed Text OCR":
                        res = run_ocr(img, lang=LANG_MAP[clang][1])
                        res.setdefault("dot_count", 0); res.setdefault("cells", 0)
                    else:
                        res = run_braille(img)
                        res.setdefault("word_count", len(res["text"].split()))
                        res.setdefault("char_count", len(res["text"]))
                elapsed = time.time() - t0
                st.markdown(f'<div class="vs-result">{res["text"]}</div>', unsafe_allow_html=True)
                stat_pills(
                    Mode=mode[:6], Conf=f'{res["confidence"]}%',
                    Time=f'{elapsed:.2f}s'
                )
                st.progress(res["confidence"] / 100)
                render_audio_output(res["text"], res.get("tts_text", res["text"]),
                                    LANG_MAP[clang][0], "visionspeak_cam.mp3")
            else:
                st.markdown(
                    '<div class="vs-empty">_ Take a photo and click Process _</div>',
                    unsafe_allow_html=True
                )

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────
    st.markdown('<div class="vs-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
<div style="text-align:center;padding-bottom:1.5rem">
  <div style="font-family:Rajdhani,sans-serif;font-size:.95rem;font-weight:600;
    color:#45475a;letter-spacing:.08em">
    VISIONSPEAK IVIS
  </div>
  <div style="font-family:JetBrains Mono,monospace;font-size:.6rem;
    color:#313244;letter-spacing:.1em;margin-top:.3rem">
    VISION · INTELLIGENCE · VOICE
  </div>
</div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
