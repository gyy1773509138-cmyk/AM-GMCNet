import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.resnet import resnet18
from dataset import get_dataloaders
from utils import save_curves, save_confusion_matrix, save_roc

dataset_path = r"C:\Users\19174\Desktop\zishiying+"
img_size, batch_size, num_classes = 150, 32, 5
num_epochs, lr = 300, 5e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, test_loader = get_dataloaders(dataset_path, img_size, batch_size)
model = resnet18(num_classes).to(device)
criterion, optimizer = nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=lr)

train_losses, test_losses, train_accs, test_accs = [], [], [], []
best_acc, best_preds, best_labels = 0, [], []
os.makedirs("result", exist_ok=True)

for epoch in range(num_epochs):
    model.train(); total, correct, loss_sum = 0, 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(); out = model(x); loss = criterion(out,y)
        loss.backward(); optimizer.step()
        loss_sum += loss.item(); _, pred = out.max(1)
        correct += (pred==y).sum().item(); total += y.size(0)
    train_losses.append(loss_sum/len(train_loader))
    train_accs.append(correct/total)

    model.eval(); total, correct, loss_sum = 0, 0, 0
    preds, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x); loss = criterion(out,y)
            loss_sum += loss.item(); _, pred = out.max(1)
            correct += (pred==y).sum().item(); total += y.size(0)
            preds.extend(pred.cpu().numpy()); labels.extend(y.cpu().numpy())
    acc = correct/total
    test_losses.append(loss_sum/len(test_loader)); test_accs.append(acc)

    if acc > best_acc: best_acc, best_preds, best_labels = acc, preds, labels
    print(f"Epoch {epoch+1}/{num_epochs}: Train Acc={train_accs[-1]:.4f}, Test Acc={acc:.4f}")

# 保存结果
np.savetxt("result/train_losses.txt", train_losses)
np.savetxt("result/test_losses.txt", test_losses)
np.savetxt("result/train_accs.txt", train_accs)
np.savetxt("result/test_accs.txt", test_accs)
save_curves(train_losses,test_losses,train_accs,test_accs,num_epochs)
save_confusion_matrix(best_labels,best_preds,num_classes)
save_roc(best_labels,best_preds,num_classes)
torch.save(model.state_dict(), "result/model.pth")
