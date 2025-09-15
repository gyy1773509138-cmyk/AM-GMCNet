# AM-GMCNet

Code for **AM-GMCNet: Galaxy Morphology Classification Network**

AM-GMCNet is a CNN-based model enhanced with **CAFM** and **AMSF** modules for galaxy morphology classification.  
This repository contains the official PyTorch implementation used in our experiments.

---

## 📂 Project Structure

- `amsf.py`, `cafm.py`, `resnet.py` —— Model components (CAFM, AMSF, ResNet backbone)
- `dataset.py` —— Data loading and preprocessing
- `utils.py` —— Helper functions (plots, confusion matrix, ROC, etc.)
- `train.py` —— Training and evaluation script
- `requirements.txt` —— Required dependencies
- `README.md` —— Project documentation

---

## 📊 Dataset

This project uses **Galaxy Zoo / SDSS DR datasets** for experiments.  
Since the dataset is large, it is **not included in this repository**.  

You can obtain the data from:  
- Galaxy Zoo: [https://data.galaxyzoo.org](https://data.galaxyzoo.org)  
- SDSS SkyServer: [https://skyserver.sdss.org](https://skyserver.sdss.org)  

After downloading, arrange the data into the following folder structure:

data/
├── Spiral/
├── Edge-on/
├── Cigar-shape/
├── Completely-round/
└── In-between-smooth/



The `train.py` script will automatically load the dataset using **torchvision.datasets.ImageFolder**.

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YourUserName/AM-GMCNet.git
   cd AM-GMCNet

python train.py --epochs 100 --batch-size 32


📈 Results

The training process will generate outputs in the result/ folder:

Loss curve (loss_curve.jpg)

Accuracy curve (accuracy_curve.jpg)

Best confusion matrix (confusion_matrix.jpg)

ROC curve (roc_curve.jpg)

Logs & metrics (.txt files)
