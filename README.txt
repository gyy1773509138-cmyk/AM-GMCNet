# AM-GMCNet: Galaxy Morphology Classification

This repository contains the PyTorch implementation of **AM-GMCNet**, a CNN-based model with CAFM and AMSF modules for galaxy morphology classification.

---

## 📂 Project Structure
- `models/` — Contains model components (`cafm.py`, `amsf.py`, `resnet.py`)
- `dataset.py` — Data loading and preprocessing
- `utils.py` — Helper functions (plots, confusion matrix, ROC)
- `train.py` — Training and evaluation script
- `requirements.txt` — Required packages

---

## 🚀 How to Run
```bash
git clone https://github.com/YourRepo/AM-GMCNet.git
cd AM-GMCNet
pip install -r requirements.txt
python train.py
