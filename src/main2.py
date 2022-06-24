import os.path
import matplotlib.pyplot as plt

import torch
import torchmetrics as metrics
from torch.utils.data import random_split

from src.model2 import *
from src.dataset import *
from torchsummary import summary
from utility.early_stopping import *


root_dir = "/Users/goktug/PycharmProjects/ML4RG Project/data"
train_filename = "train_subsequences.txt"
valid_filename = "valid_subsequences.txt"

#train_dataset = PromoterSeqDataset(root_dir, filename, transforms)
#loader = DataLoader(train_dataset, batch_size=64, collate_fn=collate_batch)

train_set = PromoterDataset(root_dir, train_filename)
valid_set = PromoterDataset(root_dir, valid_filename)

train_loader = DataLoader(train_set, batch_size=512, collate_fn=collate_batch)
valid_loader = DataLoader(valid_set, batch_size=512, collate_fn=collate_batch)

model = PromoterNet()
model = model.to(device)
summary(model, input_size=(5, 112))

loss_fn = torch.nn.MSELoss()
r2_metric_fn = metrics.R2Score()
optimizer = torch.optim.Adam(model.parameters(), lr=0.007, weight_decay=0.005)

file = open("/Users/goktug/PycharmProjects/ML4RG Project/data/train_results.txt", "w")
early_stopping = EarlyStopping(patience=5)
train_loss_list, valid_loss_list = [], []
train_r2_list, valid_r2_list = [], []

for epoch in range(70):

    model.train()
    train_loss, train_r2 = 0.0, 0.0
    valid_loss, valid_r2 = 0.0, 0.0

    print()
    for i, data in enumerate(train_loader):

        seqs, labels = data[0], data[1]
        optimizer.zero_grad()

        outputs = model(seqs)
        loss = loss_fn(outputs, labels)
        r2_value = r2_metric_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_r2 += r2_value

    ave_train_loss = train_loss / len(train_loader)
    ave_train_r2 = train_r2 / len(train_loader)
    train_loss_list.append(ave_train_loss)
    train_r2_list.append(ave_train_r2)

    print(f"Train Epoch {epoch + 1} Loss: {ave_train_loss:.4f}, R2-Score: {ave_train_r2:.4f}")
    file.write(f"Train Epoch {epoch + 1} Loss: {ave_train_loss:.4f}, R2-Score: {ave_train_r2:.4f}")

    model.eval()
    for i, data in enumerate(valid_loader):

        seqs, labels = data[0].to(device), data[1].to(device)
        outputs = model(seqs)
        loss = loss_fn(outputs, labels)
        r2_value = r2_metric_fn(outputs, labels)

        valid_loss += loss.item()
        valid_r2 += r2_value

    average_loss = valid_loss / len(valid_loader)
    early_stopping.step(average_loss)
    if early_stopping.check_patience():
        break

    ave_valid_loss = valid_loss / len(valid_loader)
    ave_valid_r2 = valid_r2 / len(valid_loader)
    valid_loss_list.append(ave_valid_loss)
    valid_r2_list.append(ave_valid_r2)

    print(f"Validation Epoch {epoch + 1} Loss: {ave_valid_loss:.4f}, R2-Score: {ave_valid_r2:.4f}")
    file.write(f"Train Epoch {epoch + 1} Loss: {ave_train_loss:.4f}, R2-Score: {ave_train_r2:.4f}")

    save_dir = "/Users/goktug/PycharmProjects/ML4RG Project/models"
    save_file_name = "model4-" + str(epoch) + ".pth"
    torch.save(model.state_dict(), os.path.join(save_dir, save_file_name))

epochs = range(1, 71)
plt.plot(epochs, train_loss_list, color="blue", label="training_loss")
plt.plot(epochs, valid_loss_list, color="red", label="validation_loss")
plt.legend()
plt.show()

plt.plot(epochs, train_r2_list, color="blue", label="training_r2")
plt.plot(epochs, valid_r2_list, color="red", label="validation_r2")
plt.legend()
plt.show()


