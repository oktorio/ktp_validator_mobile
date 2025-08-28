# 📱 KTP Validator – Mobile Version

Mobile application for **KTP validation and scoring**, built to complement the [KTP Validator backend](https://github.com/oktorio/ktp_validator).  
This version allows users to **capture or upload ID card images**, process them with AI models, and display a **Final Score (0–100)** for quality and authenticity.

---

## ✨ Features
- 📷 Capture photo directly from camera or choose from gallery  
- 📝 Validate if the uploaded file is a KTP (not another document)  
- 🎨 Detect document type:  
  - `color_document` (KTP warna)  
  - `grayscale_document`  
  - `photocopy_on_colored_background`  
  - Hard reject: `black` or `white` images  
- 🔍 Text legibility analysis (sharpness, contrast, OCR confidence, censor/blur detection)  
- 📊 Final Score (0–100) with status label (`good`, `fair`, `poor`)  

---

## 🏗️ Tech Stack
- **Frontend / Mobile:** (fill in: React Native / Flutter / Kotlin / Swift — whichever you use)  
- **Backend:** Python (Flask) [KTP Validator API](https://github.com/oktorio/ktp_validator)  
- **Libraries:** (list main libraries: OpenCV, Tesseract, TensorFlow Lite, etc.)  

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/oktorio/ktp_validator_mobile.git
cd ktp_validator_mobile
```

### 2. Install Dependencies
```bash
# Example for React Native (adjust to your framework)
npm install
```

### 3. Run the App
```bash
# Android
npx react-native run-android

# iOS
npx react-native run-ios
```

---

## 📷 Screenshots
(Will be updated soon) 

---

## 🔗 Related Projects
- [Backend – KTP Validator (Python)](https://github.com/oktorio/ktp_validator)  

---

## 📜 License
MIT License – feel free to use and modify.  

