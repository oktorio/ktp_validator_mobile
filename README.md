# KTP Validator Android (On‑Device)

A minimal Android app (Kotlin) which captures or selects a photo of an Indonesian ID (KTP) and runs **on-device analysis** to estimate document type and text legibility metrics (sharpness, edge density, RMS contrast, etc.), producing a **Final Score** 0–100.

This version implements the core heuristics of your Python `ktp_validator.py` directly in Kotlin without external native libraries, so it is easy to build and privacy-friendly (no server upload).

## Build

- Open **Android Studio** (Giraffe+ recommended).
- `File → Open…` and select the folder `KTPValidatorAndroid`.
- Let Gradle sync. Run on a device (minSdk 24).

## Use

- Tap **Take Photo** (camera) or **Pick from Gallery**.
- The app shows the image and computed metrics & score.

## Tuning

Open `ImageAnalyzer.kt`:
- `satThreshold`: what counts as "colored" saturation in HSV.
- `edgeThreshold`: Sobel magnitude threshold for edges.
- `Weights`: change feature weights and `photocopyPenalty` to mirror your Python model.
- `normalize()` bands: tighten/loosen min–max ranges to fit your dataset.

## Notes

- Implemented metrics: colored fraction, variance-of-Laplacian sharpness, Sobel edge density, RMS contrast, and a heuristic text density.
- If you prefer the **exact** Python pipeline (e.g., MSER text density, OCR), you can later **integrate OpenCV** or Tesseract via Android deps, or embed Python using Chaquopy. Start simple with this app, confirm UX and thresholds, then iterate.

## Package

- App ID: `com.vandemonium.ktpvalidator`.
- Main files:
  - `app/src/main/java/.../MainActivity.kt`
  - `app/src/main/java/.../ImageAnalyzer.kt`
  - `app/src/main/res/layout/activity_main.xml`
  - `app/src/main/AndroidManifest.xml`

---

© Vandemonium / Oktorio. For internal testing only.
