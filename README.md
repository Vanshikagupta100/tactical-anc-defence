# tactical-anc-defence
SIH 2026 (PS 26052): Real-Time AI Adaptive Noise Cancellation for Defence Communications (DCCRN + Hybrid NLMS)
# TACTICAL-ANC: Real-Time AI Adaptive Noise Cancellation
### Smart India Hackathon (SIH) 2026 | Problem Statement: 26052
**Ministry / Organization:** DRDO / iDEX / Department of Defence Production  
**Team:** SoldierMate

---

## 🎯 Overview
TACTICAL-ANC is a hybrid neural-classical speech enhancement system designed for tactical defense communications. It combines a 2.46M parameter Deep Complex Convolutional Recurrent Network (DCCRN) with a 256-tap Normalized Least Mean Squares (NLMS) filter to eliminate extreme combat noise (gunfire, artillery detonations, helicopter rotor downwash, and tank engine roar) in real-time.

---

## 🚀 Live Kaggle Training Pipeline
The complete model training pipeline, synthetic mixture generation, and validation benchmarks are hosted on Kaggle:
👉 **[Open in Kaggle Notebook](https://www.kaggle.com/code/vanshikagupta3/notebook935abd1a0b)**  
*(Trained on Tesla T4 GPU across 12,000 dynamic combat mixtures combining LibriSpeech clean speech with the Military Action Dataset [MAD]).*

---

## 🎧 Audio Demonstration sample (Click to Play)
* 🔊 **Input Audio (Combat Noise @ -5 dB SNR):** [1_original_noisy.wav](Noisy_input.wav)
* ✨ **Enhanced Voice (DCCRN + NLMS Cleaned):** [2_dccrn_boosted_voice.wav](Output_audio.wav)

## 🎧 More Audio Demonstrations across different noise levels (Listen Online Without Downloading)
### Live Interactive Players on Kaggle (Zero-Download)
Listen to the model clean authentic combat speech directly inside the interactive notebook:
👉 **[Click Here to Listen on Kaggle (Scroll to Cell 11)](https://www.kaggle.com/code/vanshikagupta3/notebook935abd1a0b#135402774)**

---

## 🛠️ Repository Contents
* `SIH-2026-Tactical-ANC-DCCRN.ipynb`: Complete Kaggle training notebook with dataset pipeline and complex loss functions.
* `record_and_clean.py`: Standalone Python script for real-time microphone capture and live audio denoising.
* `dccrn_weights.pt`: Pre-trained PyTorch model weights (165 layers, ~9.9 MB).
* `hybrid_adaptive_filter.py`: 256-tap Normalized Least Mean Squares (NLMS) acoustic post-filter.
