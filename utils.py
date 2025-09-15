import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import torch

def save_curves(train_losses, test_losses, train_accs, test_accs, num_epochs, out_dir="result"):
    plt.figure()
    plt.plot(range(1, num_epochs+1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs+1), test_losses, label="Test Loss")
    plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss Curve")
    plt.savefig(f"{out_dir}/loss_curve.jpg")

    plt.figure()
    plt.plot(range(1, num_epochs+1), train_accs, label="Train Acc")
    plt.plot(range(1, num_epochs+1), test_accs, label="Test Acc")
    plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy Curve")
    plt.savefig(f"{out_dir}/accuracy_curve.jpg")

def save_confusion_matrix(labels, preds, num_classes, out_dir="result"):
    cm = confusion_matrix(labels, preds)
    plt.figure(); plt.imshow(cm, cmap="Blues"); plt.colorbar(); plt.title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.savefig(f"{out_dir}/confusion_matrix.jpg")

def save_roc(labels, preds, num_classes, out_dir="result"):
    plt.figure()
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(np.array(labels)==i, np.array(preds)==i)
        plt.plot(fpr, tpr, label=f"Class {i}, AUC={auc(fpr,tpr):.2f}")
    plt.plot([0,1],[0,1],"k--")
    plt.legend(); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve")
    plt.savefig(f"{out_dir}/roc_curve.jpg")
