import torch
from torch import nn

import torchvision #tv.datasets.MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import random

train_data = torchvision.datasets.MNIST(
    "data",
    train=True,
    transform=ToTensor(),
    download=False,
)

test_data = torchvision.datasets.MNIST(
    "data",
    train=False,
    transform=ToTensor(),
    download=False
)

'''# print(len(train_data), len(test_data))


# plt.figure(figsize=(9, 9))
# for i in range(5):
#     image = train_data[i][0] # sample [image, label]
#     image = image.squeeze()
#     label = train_data[i][1]
#     plt.subplot(2, 3, i+1)
#     plt.imshow(image, cmap='gray')
#     plt.title(train_data.classes[label])
#     plt.axis(False)
# plt.tight_layout()
# plt.show()
'''

train_data_loader = DataLoader(train_data,
                            batch_size=32,
                            shuffle=True)

test_data_loader = DataLoader(test_data,
                              batch_size=32,
                              shuffle=False)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#to get shape of data
class MNISTModel(nn.Module):
    def __init__(self, inp, hidden, out):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(inp, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classification = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden*7*7, out_features=out) #*7*7 because of compression, idk if its similar for all models
            # 7*7 produced mis shape error
        )

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classification(x)
        return x
    
model_2 = MNISTModel(1, 10, 10)
model_2.to(device=DEVICE)

from timeit import default_timer as timer
from helper_functions import accuracy_fn
from tqdm.auto import tqdm

#training T-T
# loss, optim, time, acc
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_2.parameters(), lr=0.1)

torch.manual_seed(42)
start_time = timer()
epochs = 1

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-------")
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(train_data_loader):
        model_2.train()
        X, y = X.to(DEVICE), y.to(DEVICE)
        train_logits = model_2(X)
        train_probs = torch.softmax(train_logits, dim=1)
        train_labels = torch.argmax(train_probs)
    
        loss = loss_fn(train_logits, y)
        acc = accuracy_fn(y_true=y, y_pred=train_logits.argmax(dim=1))

        train_loss += loss
        train_acc += acc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_data_loader)
    # train_acc /= len(train_data_loader)

    test_loss, test_acc = 0, 0
    model_2.eval()
    with torch.inference_mode():
        for batch, (X, y) in enumerate(test_data_loader):
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred_logits = model_2(X)
            pred_probs = torch.softmax(pred_logits, dim=1)
            pred_labels = torch.argmax(pred_probs)

            test_loss += loss_fn(pred_logits, y)
            test_acc += accuracy_fn(y_true=y, y_pred=pred_logits.argmax(dim=1))
    
        test_loss /= len(test_data_loader) 
        test_acc /= len(test_data_loader) 

    print(f"Train Loss: {train_loss:.2f} | Test loss: {test_loss:.2f} | Test Acc: {test_acc:.2f}%")

end_time = timer()

print(f"Total time taken: {end_time - start_time:.2f} seconds")

import numpy as np

random_list = np.arange(0, 10, 1)
print(random_list)
model_2.eval()
fig = plt.figure(figsize=(12, 12))
with torch.inference_mode():
    for i in random_list:
        image = test_data[i][0].to(DEVICE)
        label = test_data[i][1]

        y_logits = model_2(image.unsqueeze(dim=0))
        y_probs = torch.softmax(y_logits, dim=1)
        y_labels = torch.argmax(y_probs, dim=1)
        
        fig.add_subplot(4, 4, i+1)
        plt.imshow(image.squeeze().cpu(), cmap='gray')
        if y_labels.item() == label:
            plt.title(f"Truth: {test_data[i][1]} | Pred: {y_labels.item()}", c="g")
        else:
            plt.title(f"Truth: {test_data[i][1]} | Pred: {y_labels.item()}", c="r")
        plt.axis(False)
    plt.tight_layout()
    plt.show()




# import numpy as np
# plt.figure(figsize=(9, 9))
# for i in np.random.randint(0, len(test_data_loader), 5):
#     image = test_data_loader[i][0]
#     image = image.squeeze()
#     label = test_data_loader[i][1]
#     plt.subplot(2, 3, i+1)
#     plt.imshow(image, cmap='gray')



