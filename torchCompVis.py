import torch
from torch import nn

from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

train_data = datasets.FashionMNIST("data", 
                                   train=True,
                                   transform=ToTensor(),
                                   download=False)

test_data = datasets.FashionMNIST("data",
                                  train=False,
                                  transform=ToTensor(),
                                  download=False)

# print(len(train_data.data), len(train_data.targets))
# print(len(test_data.data), len(test_data.targets))

# print(train_data.classes)

# image, label = train_data[0]
# plt.imshow(image.squeeze(), cmap="gray")
# plt.title(train_data.classes[label])
# plt.show()

# torch.manual_seed(42)
# fig = plt.figure(figsize=(9, 9))
# rows, cols = 4, 4
# for i in range(1, rows * cols + 1):
#     random_num = torch.randint(0, len(train_data), size=[1]).item()
#     img, label = train_data[random_num]
#     fig.add_subplot(rows, cols, i)
#     plt.imshow(img.squeeze(), cmap="gray")
#     plt.title(train_data.classes[label])
#     plt.axis(False)

# plt.tight_layout()
# plt.show()

#NOW LOADING THE DATA AFTER GENERATION
BATCH_SIZE = 32

train_loaded_data = DataLoader(train_data,
                               batch_size=BATCH_SIZE,
                               shuffle=True)

test_loaded_data = DataLoader(test_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

# print(f"Size of each batch: {len(train_loaded_data)} and {len(test_loaded_data)}")
# parsing train data loader for info

# train_data_features, train_data_labels = next(iter(train_loaded_data))

# torch.manual_seed(42)
# random_int = torch.randint(0, len(train_data_features), size=[1]).item()
# img, label = train_data_features[random_int], train_data_labels[random_int]
# plt.imshow(img.squeeze(), cmap="gray")
# plt.title(train_data.classes[label])
# plt.show()

#BUILDING FIRST MODEL
classes = train_data.classes

# class FashionMNISTModelV0(nn.Module):
#     def __init__(self, in_shape, out_shape, hidden_uni):
#         super().__init__()
#         self.linear_layer_stack = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(in_features=in_shape, out_features=hidden_uni),
#             nn.Linear(in_features=hidden_uni, out_features=out_shape)
#         )

#     def forward(self, x):
#         return self.linear_layer_stack(x)
    
# torch.manual_seed(42)

# model_0 = FashionMNISTModelV0(in_shape=784,
#                               out_shape=len(classes),
#                               hidden_uni=10)

# model_0.to("cpu")

#importing accuracy func from helper file
from helper_functions import accuracy_fn

# loss_fn = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

# creating timing function
from timeit import default_timer as timer
def print_train_time(start, end, device=None):
    total_time = end-start
    print(f"Train Time on {device}: {total_time:.3f} sec")
    return total_time

#NOW TRAINING THE MODEL

from tqdm.auto import tqdm #for progress

# torch.manual_seed(42)
# train_time_start_on_cpu = timer() # start the timer

# epochs = 3 

# for epoch in tqdm(range(epochs)):
#     print(f"Epoch: {epoch}\n-----")
#     train_loss = 0
#     for batch, (X, y) in enumerate(train_loaded_data):
#         model_0.train()
#         y_pred = model_0(X)
#         loss = loss_fn(y_pred, y)
#         train_loss += loss # adding to main loss param

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if batch % 400 == 0: # total batch size is 1800 something
#             print(f"Looked at {batch * len(X)}/{len(train_loaded_data.dataset)} samples")

#     train_loss /= len(train_loaded_data)
#     #testing
#     test_loss, test_acc = 0, 0
#     model_0.eval()
#     with torch.inference_mode():
#         for X, y in test_loaded_data:
#             test_pred = model_0(X)
#             test_loss += loss_fn(test_pred, y)
#             test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

#         test_loss /= len(test_loaded_data)
#         test_acc /= len(test_loaded_data)

#     print(f"\nTrain Loss: {train_loss:.5f} | Test Loss: {test_loss:.2f} | Train Acc: {test_acc:.2f}%")

# end time for loop
# train_time_end_on_cpu = timer()
# total_train_time_model_0 = print_train_time(train_time_start_on_cpu,
#                                              train_time_end_on_cpu,
#                                              device=str(next(model_0.parameters()).device))

#MAKING PREDICTIONS
# torch.manual_seed(42)
def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device: torch.device = None):
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
        
        loss /= len(data_loader)
        acc /= len(data_loader)

    return {"model_name": model.__class__.__name__,
            "model_loss": loss.item(),
            "model_acc": acc}

# # model_0_results = eval_model(model_0, test_loaded_data, loss_fn, accuracy_fn)
# # print(model_0_results)

device = "cuda" if torch.cuda.is_available() else "cpu"

# #BUILDING ANOTHER MODEL -> NON-LINEAR MODEL

# class FashionMNISTModelV1(nn.Module):
#     def __init__(self, inp, hidden, out):
#         super().__init__()
#         self.nonLinear_layer_stack = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(in_features=inp, out_features=hidden),
#             nn.ReLU(),
#             nn.Linear(in_features=hidden, out_features=out),
#             nn.ReLU()
#         )

#     def forward(self, x: torch.Tensor):
#         return self.nonLinear_layer_stack(x)

# #preparing the model
# model_1 = FashionMNISTModelV1(inp=784, hidden=10,
#                               out=len(classes)).to(device)

# # loss, optim, acc
# loss_fn = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.1)


# # train/test loop
def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    train_loss, train_acc = 0, 0
    model.to(device)
    model.train()
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        train_pred = model(X)
        loss = loss_fn(train_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y, y_pred=train_pred.argmax(dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # calc loss and acc
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.2f}%")

def test_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device: torch.device = device):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            test_pred = model(X)
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
    
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test Loss: {test_loss:.2f} | Test Acc: {test_acc:.2f}%\n")
    
# torch.manual_seed(42)
# start_time = timer()
# epochs = 3
# for epoch in tqdm(range(epochs)):
#     print(f"Epoch: {epoch}\n------")
#     train_step(model_1,
#                train_loaded_data,
#                loss_fn,
#                optimizer,
#                accuracy_fn,
#                device)
#     test_step(model_1,
#               test_loaded_data,
#               loss_fn,
#               accuracy_fn,
#               device
#               )

# end_time = timer()
# total_train_time = print_train_time(start_time, end_time, device)

# model_1_eval = eval_model(model_1, test_loaded_data, loss_fn, accuracy_fn, device)
# print(model_1_eval)

#MODEL 2 USING CNNs

class FashionMNISTModelV2(nn.Module):
    def __init__(self, inp, out, hidden):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=inp,
                      out_channels=hidden,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden,
                      out_channels=hidden,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden*7*7, out_features=out)
        )

    def forward(self, x: torch.tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x
    
torch.manual_seed(42)
model_2 = FashionMNISTModelV2(1, len(classes), 10).to(device)

#creating toy data for above model testing

# torch.manual_seed(42)
# images = torch.randn(size=(32, 3, 64, 64))
# test_image = images[0]
# print(f"Image batch shape: {images.shape}")
# print(f"Single image shape: {test_image.shape}")
# print(f"Single Image pixel Values: \n{test_image}")

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_2.parameters(), lr=0.1)

torch.manual_seed(42)

start_time = timer()
epochs = 3

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")
    train_step(model_2, train_loaded_data, loss_fn, optimizer, accuracy_fn, device)
    test_step(model_2, train_loaded_data, loss_fn, accuracy_fn, device)

end_time = timer()
total = print_train_time(start_time, end_time, device)

#eval model 2
# model_2_results = eval_model(
#     model_2,
#     test_loaded_data,
#     loss_fn,
#     accuracy_fn,
#     device
# )

# NOW MAKING PREDS FUNCTION
def make_predictions(model:torch.nn.Module,
                     data: list,
                     device: torch.device = device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0).to(device)
            pred_logit = model(sample)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)
            pred_probs.append(pred_prob)
        
    return torch.stack(pred_probs)

import random

random.seed(42)
test_samples = []
test_labels = []
for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)

pred_probs = make_predictions(model_2, test_samples, device)

pred_classes = pred_probs.argmax(dim=1)

#plotting preds 
plt.figure(figsize=(9, 9))
nrows = 3
ncols = 3

for i, sample in enumerate(test_samples):
    plt.subplot(nrows, ncols, i+1)
    plt.imshow(sample.squeeze(), cmap="gray")
    pred_label = classes[pred_classes[i]]
    truth_label = classes[test_labels[i]]
    title_text = f"Pred: {pred_label} | Truth: {truth_label}"

    if pred_label == truth_label:
        plt.title(title_text, c='g', fontsize=10)
    else:
        plt.title(title_text, c='r', fontsize=10)
    plt.axis(False)
plt.show()