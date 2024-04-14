# import torch
# from torch import nn

#scalar 
# scalar = torch.tensor(7)
# scalar.ndim -> dimensions => 0
# scalar.item() -> gives value => 7

#vector
# vec = torch.tensor([7, 7])
# print(vec.ndim) #-> 1 dim
# print(vec.shape)

#MATRIX
# MATRIX = torch.tensor([
#     [1, 2],
#     [3, 4]
# ])
# MATRIX shape is [2, 2] 2 wide and 2 high

#tensor
# tensor = torch.tensor([[[1,2,3], [4,5,6]]])
# tensor shape => 3
# print(tensor.shape) => [1, 2, 3] shape


#random tensors generation
# randomTensors = torch.rand(size=(3, 4))
# print(randomTensors, randomTensors.dtype)
#
# torch.zeros(size=(3, 4))
# torch.ones(size=(3, 4))
#
# serialTensor = torch.arange(start=0, end=10, step=1)
# print(serialTensor.dtype)
#
# zeroTensor = torch.zeros_like(input=someTensorWithSomeShape) -> generates 0 filled tensor with similar shape
# oneTensor = torch.ones_like....

#Tensor data types
# floatTensor = torch.tensor([1.0,2.0,3.0], dtype=None, device=None, requires_grad=False) -> cpu based float32 tensor
# print(floatTensor.dtype, floatTensor.device, floatTensor.shape)
# common dtypes (usage: torch.[dtype]) -> float32/16/64 float/half/double

#getting info from tensors
# tensor = torch.rand(size=(3, 4))
#
# print(tensor)
# print("Shape of the tensor: ", tensor.shape)
# print("Datatype of the tensor: ", tensor.dtype)
# print("Device tensor is stored on: ", tensor.device)

#operations on tensors
# tensor = torch.tensor([1,2,3])
# tensor *= 10 #'''Reassign tensors'''
# print(tensor)
# more operations like => torch.mul(tensor, 10) or torch.add(tensor, 10)

#MATRIX MULTIPLICATION
# tensor = torch.tensor([1,2,3])
#
# print(tensor * tensor) -> element wise multiplication
# print(torch.matmul(tensor, tensor)) -> matrix multiplication
# print(tensor @ tensor) => not recommended
#
# print(torch.matmul(tensor, tensor))

# ~~~~~~ AVOID SHAPE ERRORS ~~~~~~ *(same inners)*

# TO MUL MATRICES OF SAME DIMENSION WE CAN USE TRANSPOSE METHOD TO MODIFY ITS SHAPE FOR THE OPERATION
#
# tensorA = torch.rand(2, 3)
# tensorB = torch.rand(2, 3)
#
# print(tensorA)
# print(tensorB)
# 
# print(torch.matmul(tensorA, tensorB.T)) -> torch.mm() => short for matmul

# tensor_A = torch.rand(size=(2, 3))

# torch.manual_seed(42)

# linear = torch.nn.Linear(in_features=3,
#                          out_features=2)

# output = linear(tensor_A)
# print(f"output: {output}")
# print(f"output shape: {output.shape}")

#FINDING AGGREGATORS (MIN, MAX, MEAN, SUM, ETC..)
# tensor = torch.arange(start=10, end=110, step=10)
#
# print(tensor)
# print(f"Minimum: {tensor.min()}")
# print(f"Maximum: {tensor.max()}")
# print(f"Sum: {tensor.sum()}")
# print(f"Mean: {tensor.type(torch.float32).mean()}") // torch.[method_name] -> min, max, mean, etc.

#INDEX OF MIN/MAX
# tensor = torch.arange(10, 100, 10)
# print(f"tensor => {tensor}")
# print(f"Index for above min val: {tensor.argmin()}")
# print(f"Index for above max val: {tensor.argmax()}")

# to change dtype of any tensor we can also use => tensor.type(torch.[dtype])

# torch.reshape(input, shape)	Reshapes input to shape (if compatible), can also use torch.Tensor.reshape().
# Tensor.view(shape)	Returns a view of the original tensor in a different shape but shares the same data as the original tensor.
# torch.stack(tensors, dim=0)	Concatenates a sequence of tensors along a new dimension (dim), all tensors must be same size.
# torch.squeeze(input)	Squeezes input to remove all the dimenions with value 1.
# torch.unsqueeze(input, dim)	Returns input with a dimension value of 1 added at dim.
# torch.permute(input, dims)	Returns a view of the original input with its dimensions permuted (rearranged) to dims.

#INDEXING
# tensor = torch.arange(1, 10, 1).reshape(1, 3, 3)

# print(f"First(toppest): {tensor[0]}")
# print(f"Second: {tensor[0][0]}")
# print(f"Third(deepest): {tensor[0][0][0]}")

# different ways of parsing

# print(tensor[:, :, 0]) all of 0th dimension -> all of 1st dimension -> 0th index(first ele) of 2nd dimension

#PYTORCH TENSORS AND NUMPY
# numpy to tensor
# import numpy as np

# arr = np.arange(1.0, 8.0)
# tensor = torch.from_numpy(arr).type(torch.float32)

# print(arr)
# print(tensor)
# print(tensor.dtype)

# tensor to numpy arr
# tensor = torch.arange(1, 10, 1)
# numArr = tensor.numpy()

# print(tensor)
# print(type(numArr))

#RANDOMNESS IN RANDOM

# RANDOM_SEED = 42

# torch.manual_seed(RANDOM_SEED)
# tensorA = torch.rand(3, 4)

# torch.manual_seed(RANDOM_SEED)
# tensorB = torch.rand(3, 4)

# print(tensorA)
# print(tensorB)
# print(tensorA == tensorB)

#GPU ENABLED COMPUTING
# device = "cuda" if torch.cuda.is_available() else "cpu"

# print(torch.cuda.device_count())
# tensor = torch.rand(size=(3, 2))
# print(tensor, tensor.device)

# tensor = tensor.to(device=device)
# print(tensor.device)
# use tensorOnGPU.cpu() -> to get tensor back on gpu (for numpy ops can't use gpu)

#~~~~~~~~~EXERCISE~~~~~~~~~~

#2
# tensor = torch.rand(size=(7, 7))
# print(tensor.shape)

#3
# tensor3 = torch.rand(size=(1, 7))
# print(torch.matmul(tensor, tensor3.T))

#4
# torch.manual_seed(0)
# tensor_seed_01 = torch.rand(size=(7, 7))

# torch.random.manual_seed(0)
# tensor_seed_02 = torch.rand(size=(1, 7))

# print(tensor_seed_01)
# print(tensor_seed_02)
# print(f"Multiplication of matrix: \n {torch.matmul(tensor_seed_01, tensor_seed_02.T)}")

#5
# device = "cuda" if torch.cuda.is_available() else "cpu"

# torch.cuda.manual_seed(0)
# tensor1 = torch.rand(size=(2, 2), device=device)

# torch.cuda.manual_seed(0)
# tensor2 = torch.rand(size=(2, 2), device=device)

# print(tensor1)
# print(tensor2)
# print(tensor1 == tensor2)

#6
# torch.manual_seed(1234)
# tensorOne = torch.rand(size=(2, 3))
# tensorTwo = torch.rand(size=(2, 3))

# tensorOne = tensorOne.to(device=device)
# tensorTwo = tensorTwo.to(device=device)

# print(tensorOne, tensorOne.device)
# print(tensorTwo, tensorTwo.device)
# mul = torch.matmul(tensorOne, tensorTwo.T)
# print(mul, mul.shape, mul.dtype)
# print(f"min value of matmul: {mul.min()}")
# print(f"max value of matmul: {mul.max()}")

#7, 8, 9 done above
#10

# torch.manual_seed(7)
# tensorRand = torch.rand(size=(1, 1, 1, 10))

# tensorOne = tensorRand.squeeze()

# print(tensorRand, tensorRand.shape)
# print(tensorOne, tensorOne.shape)


# ~~~~~~~ML WORKFLOW USING PYTORCH~~~~~~~~~
# import matplotlib.pyplot as plt
# #BUILDING(PREPARING) DATA FOR TRAINING

# # y = x.m + b -> x is linear dataset, m is weight, b is bias

# weight = 0.7 #m
# bias = 0.3 #b

# start = 0
# end = 1
# step = 0.02
# X = torch.arange(start, end, step).unsqueeze(dim=1)
# #whole above is x
# y = X * weight + bias

# #DATA SPLITTING FOR TRAINING AND TESTING
# train_split = int(0.8 * len(X)) # 80% train split 20% test split
# X_train, y_train = X[:train_split], y[:train_split]
# X_test, y_test = X[train_split:], y[train_split:]

# # print(len(X_train), len(y_train), len(X_test), len(y_test)) => outs => 40 40 10 10

# #plotting data (Visualise)
# def plotting_data(train_data=X_train,
#                   train_labels=y_train,
#                   test_data=X_test,
#                   test_labels=y_test,
#                   predictions=None):
#     plt.figure(figsize=(10, 7))
#     plt.scatter(train_data, train_labels, c="b", s=4, label="TrainingData")
#     plt.scatter(test_data, test_labels, c="g", s=4, label="TestingData")
#     if predictions is not None:
#         plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
    
#     plt.legend(prop={"size": 14})
#     plt.waitforbuttonpress()

# # plotting_data()

# #BUILDING MODEL
# class LinearRegressionModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
#         self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)

#     def forward(self, x : torch.Tensor) -> torch.Tensor:
#         return self.weights * x + self.bias
    
#GETTING CONTENT FROM NN MODEL
# torch.manual_seed(42)
# model_0 = LinearRegressionModel()
# print(model_0.state_dict()) #can also use parameters within list method

# with torch.inference_mode():
#     y_preds = model_0(X_test)

# print(f"Length of sample test: {len(X_test)}")
# print(f"Length of predictions made: {len(y_preds)}")
# print(y_preds)

#Plotting preds
# plotting_data(predictions=y_preds)

#TRAIN MODEL FOR BETTER PREDICTION
#CREATING LOSS FUNC AND OPTIMIZER

# loss_fn = nn.L1Loss()
# optimizer = torch.optim.SGD(model_0.parameters(), lr=0.01)

#CREATING LEARNING AND TESTING LOOP
# torch.manual_seed(42)
# epochs = 100

# train_loss_values = []
# test_loss_values = []
# epoch_count = []

# for epoch in range(epochs):
#     model_0.train() # in train mode
#     y_pred = model_0(X_train) # forward pass(make preds)
#     loss = loss_fn(y_pred, y_train) # calc loss
#     optimizer.zero_grad() # put optimizer in zero grad for each pass
#     loss.backward() # back propagate the loss (nhi aaya smjh)
#     optimizer.step() # update

#     #testing
#     model_0.eval() # put in test mode
#     with torch.inference_mode(): # turn of training features
#         test_pred = model_0(X_test) # get preds
#         test_loss = loss_fn(test_pred, y_test.type(torch.float)) # calc loss in preds

#         if epoch % 10 == 0:
#             epoch_count.append(epoch)
#             train_loss_values.append(loss.detach().numpy())
#             test_loss_values.append(test_loss.detach().numpy())
#             print(f"Epoch: {epoch} | MAE train loss: {loss} | MAE test loss: {test_loss}")

# plt.plot(epoch_count, train_loss_values, label="Train Loss")
# plt.plot(epoch_count, test_loss_values, label="Test Loss")
# plt.title("Test and Train Loss curves")
# plt.xlabel("loss")
# plt.ylabel("epoch_cnt")
# plt.legend()
# plt.waitforbuttonpress()
            
# model_0.eval()
# with torch.inference_mode():
#     y_pred = model_0(X_test)
# plotting_data(predictions=y_pred)
            
#SAVING TRAINED AND TESTED MODEL
# from pathlib import Path

# # create model directory
# MODEL_PATH = Path('GGs/models')
# # MODEL_PATH.mkdir(parents=True, exist_ok=True)

# # create model save path(path/name)
# MODEL_NAME = "01_pytorch_workflow_model_0.pth"
# MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# # saving the model
# # print(f"Saving the model at: {MODEL_SAVE_PATH}")
# # torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)

# loaded_model_0 = LinearRegressionModel()
# loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

# loaded_model_0.eval()
# with torch.inference_mode():
#     preds_loaded_model = loaded_model_0(X_test)

# plotting_data(predictions=preds_loaded_model)

# import torch
# from torch import nn
# import matplotlib.pyplot as plt
# from pathlib import Path

# # if cuda available
# device = "cuda" if torch.cuda.is_available() else "cpu"

# #NEED DATA -> MODEL -> TRAIN/TEST -> SAVE -> LOAD -> PREDS
# start = 0
# end = 1
# step = 0.02

# weights = 0.7
# bias = 0.3

# X = torch.arange(start, end, step).unsqueeze(dim=1)
# y = weights * X + bias

# # 80/20
# train_split = int(0.8 * len(X))
# X_train, y_train = X[:train_split], y[:train_split]
# X_test, y_test = X[train_split:], y[train_split:]

# def plotting_data(train_data=X_train,
#                   train_labels=y_train,
#                   test_data=X_test,
#                   test_labels=y_test,
#                   predictions=None):
#     plt.figure(figsize=(10, 7))
#     plt.scatter(train_data, train_labels, c="b", s=4, label="Training Data")
#     plt.scatter(test_data, test_labels, c="g", s=4, label="Testing Data")
#     if predictions is not None:
#         plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
#     plt.legend(prop={"size": 14})
#     plt.waitforbuttonpress()

# # creating model
# class LinearRegressionModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear_layer = nn.Linear(in_features=1, out_features=1)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.linear_layer(x)

# torch.manual_seed(42)
# model_1 = LinearRegressionModel()
# model_1.to(device)

# print(next(model_1.parameters()).device)

# # plotting_data(predictions=y_preds)
    
# #TRAIN AND TEST 
# # loss_fn = torch.nn.L1Loss()
# # optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.01)
# # creating train/test loop
# # epochs = 1000

# torch.manual_seed(42)
# X_train = X_train.to(device)
# y_train = y_train.to(device)
# X_test = X_test.to(device)
# y_test = y_test.to(device)

# # for epoch in range(epochs):
# #     model_1.train()
# #     y_preds = model_1(X_train)
# #     loss = loss_fn(y_preds, y_train)
# #     optimizer.zero_grad()
# #     loss.backward()
# #     optimizer.step()

# #     #testing
# #     model_1.eval()
    
# #     with torch.inference_mode():
# #         test_preds = model_1(X_test)
# #         test_loss = loss_fn(test_preds, y_test)

#         # if epoch % 100 == 0:
#         #     print(f"Epoch: {epoch} | Train Loss: {loss} | Test Loss: {test_loss}")

# # print(f"Models state: {model_1.state_dict()}")
# # print(f"Actual -> weight : {weights} | bias: {bias}")
# # model_1.eval()
# # with torch.inference_mode():
# #     y_preds = model_1(X_test)

# # plotting_data(predictions=y_preds.cpu())

# MODEL_PATH = Path("GGs/models")
# MODEL_PATH.mkdir(parents=True, exist_ok=True)

# MODEL_NAME = "02_pytorch_workflow_model_1.pth"
# MODEL_SAVE_PATH = MODEL_PATH/MODEL_NAME

# # torch.save(obj=model_1.state_dict(), f=MODEL_SAVE_PATH)

# loaded_model_1 = LinearRegressionModel()
# loaded_model_1.load_state_dict(torch.load(MODEL_SAVE_PATH))
# loaded_model_1.to(device)

# loaded_model_1.eval()
# with torch.inference_mode():
#     loaded_preds = loaded_model_1(X_test)

# plotting_data(predictions=loaded_preds.cpu())
# import torch
# from torch import nn
# import matplotlib.pyplot as plt

# weight = 0.3
# bias = 0.9

# start = 0
# end = 1
# step = 0.01
# X = torch.arange(start, end, step).unsqueeze(dim=1)
# y = weight * X + bias

# train_split = int(0.8 * len(X))
# x_train, y_train = X[:train_split], y[:train_split]
# x_test, y_test = X[train_split:], y[train_split:]

# def plotting_data(trainD=x_train,
#                   trainL=y_train,
#                   testD=x_test,
#                   testL=y_test,
#                   predictions=None):
#     plt.figure(figsize=(10, 7))
#     plt.scatter(trainD, trainL, c="b", s=4, label="Training Data")
#     plt.scatter(testD, testL, c="g", s=4, label="Testing Data")

#     if predictions is not None:
#         plt.scatter(testD, predictions, c="r", s=4, label="Predictions")
#     plt.show()

# device = "cuda" if torch.cuda.is_available() else "cpu"

# class LRM(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.weight = nn.Parameter(torch.randn(1, dtype=torch.float, requires_grad=True))
#         self.bias = nn.Parameter(torch.randn(1, dtype=torch.float, requires_grad=True))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.weight * x + self.bias

# torch.manual_seed(42)
# model_0 = LRM()
# model_0.to(device=device)

# loss_fn = torch.nn.L1Loss()
# optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

# epochs = 300

# torch.manual_seed(42)
# x_train = x_train.to(device)
# y_train = y_train.to(device)
# x_test = x_test.to(device)
# y_test = y_test.to(device)

# for epoch in range(epochs):
#     model_0.train()
#     y_preds = model_0(x_train)
#     loss = loss_fn(y_preds, y_train)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     model_0.eval()
#     with torch.inference_mode():
#         test_preds = model_0(x_test)
#         test_loss = loss_fn(test_preds, y_test)

#     # if epoch % 20 == 0:
#     #     print(f"Epoch: {epoch} | Training Loss: {loss} | Test Loss: {test_loss}")

# with torch.inference_mode():
#     y_preds = model_0(x_test)

# # plotting_data(predictions=y_preds.cpu())
# from pathlib import Path

# MODEL_PATH = Path("GGs/models")
# MODEL_PATH.mkdir(parents=True, exist_ok=True)

# MODEL_NAME = "03_pytorch_workflow_model_0.pth"
# MODEL_SAVE_NAME = MODEL_PATH/MODEL_NAME

# # torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_NAME)
# # print(f"Model saved at: {MODEL_SAVE_NAME}")

# loaded_model_1 = LRM()
# loaded_model_1.load_state_dict(torch.load(MODEL_SAVE_NAME))
# loaded_model_1.to(device)

# with torch.inference_mode():
#     load_preds = loaded_model_1(x_test)

# plotting_data(predictions=load_preds.cpu())

#~~~~~~~~~~~~CLASSIFICATION PROBLEM~~~~~~~~~~~~~

#PREPARING DATA

# from sklearn.datasets import make_circles
# from sklearn.model_selection import train_test_split
# import pandas as pd
# import matplotlib.pyplot as plt
# import torch
# from torch import nn
# from helper_functions import plot_predictions, plot_decision_boundary

# n_samples = 1000

# x, y = make_circles(n_samples, noise=0.03, random_state=42)

# circles = pd.DataFrame({
#     "X1": x[:, 0],
#     "X2": x[:, 1],
#     "Label(y)": y
# })

# plt.scatter(x=x[:, 0],
#             y=x[:, 1],
#             c=y,
#             cmap=plt.cm.RdYlBu)

# plt.show()

# x = torch.from_numpy(x).type(torch.float)
# y = torch.from_numpy(y).type(torch.float)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# print(len(x_train), len(x_test), len(y_train), len(y_test))

# def plotting_data(train_data=x_train,
#                   train_label=y_train,
#                   test_data=x_test,
#                   test_label=y_test,
#                   predictions=None):
#     fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#     axs[0].scatter(train_data[:, 0], train_data[:, 1], c=y_train, cmap=plt.cm.RdYlBu)
#     axs[1].scatter(test_data[:, 0], test_data[:, 1], c=y_test, cmap=plt.cm.bwr)

#     plt.show()

# plotting_data()
#BULDING MODEL

# device = "cuda" if torch.cuda.is_available() else "cpu"

# class CircleModelV0(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer_1 = nn.Linear(in_features=2, out_features=5)
#         self.layer_2 = nn.Linear(in_features=5, out_features=1)

#     def forward(self, x):
#         return self.layer_2(self.layer_1(x))

# model_0 = nn.Sequential(
#     nn.Linear(in_features=1, out_features=5),
#     nn.Linear(in_features=5, out_features=1)
# ).to(device)
# model_0 = CircleModelV0()
# model_0.to(device)

# untrained_preds = model_0(x_test.to(device))
 
# above is not producing same out as y_test
# we need to modify it

# LOSS FUNCTION AND OPTIMIZER
# loss_fn = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

#introducing accuracy metric
# def accuracy_fn(y_test, y_pred):
#     correct = torch.eq(y_test, y_pred).sum().item()
#     acc = ( correct / len(y_pred) ) * 100 #time 100 for percent
#     return acc
# RAW MODEL OUT TO PRED LABELS: LOGITS -> PRED_PROBS -> PRED_LABELS(Y)

# y_logits = model_0(x_test.to(device))[:5]
# y_pred_probs = torch.sigmoid(y_logits)

# y_preds = torch.round(y_pred_probs)


# print(y_preds.squeeze())

#BUILDING A TRAINING AND TESTING LOOP
# torch.manual_seed(42)

# epochs = 100
# x_train, y_train = x_train.to(device), y_train.to(device)
# x_test, y_test = x_test.to(device), y_test.to(device)

# for epoch in range(epochs):
#     model_0.train()

#     y_logits = model_0(x_train).squeeze()
#     y_preds = torch.round(torch.sigmoid(y_logits))

#     loss = loss_fn(y_logits, y_train)
#     acc = accuracy_fn(y_test=y_train,
#                       y_pred=y_preds)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     # testing
#     model_0.eval()
#     with torch.inference_mode():
#         test_logits = model_0(x_test).squeeze()
#         test_preds = torch.round(torch.sigmoid(test_logits))

#         test_loss = loss_fn(test_logits, y_test)
#         test_acc = accuracy_fn(y_test=y_test, y_pred=test_preds)

#         # if epoch % 10 == 0:
#         #     print(f"Epoch: {epoch} | Train Loss: {loss} | Test Loss: {test_loss}")
#         #     print(f"Train Accuracy: {acc} | Test Accuracy: {test_acc}")

# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.title("Train")
# plot_decision_boundary(model_0, x_train, y_train)
# plt.subplot(1, 2, 2)
# plt.title("Test")
# plot_decision_boundary(model_0, x_test, y_test)
# plt.show()

# class CircleModelV2(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer_1 = nn.Linear(in_features=2, out_features=10)
#         self.layer_2 = nn.Linear(in_features=10, out_features=10)
#         self.layer_3 = nn.Linear(in_features=10, out_features=1)
#         self.relu = nn.ReLU()
    
#     def forward(self, x):
#         return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

# model_3 = CircleModelV2()
# model_3.to(device)

# loss_fn = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.SGD(params=model_3.parameters(), lr=0.1)

# torch.manual_seed(42)

# epochs = 1000

# x_train, y_train = x_train.to(device), y_train.to(device)
# x_test, y_test = x_test.to(device), y_test.to(device)

# for epoch in range(epochs):
#     model_3.train()
#     y_logits = model_3(x_train).squeeze()
#     y_preds = torch.round(torch.sigmoid(y_logits))

#     loss = loss_fn(y_logits, y_train)
#     acc = accuracy_fn(y_test=y_train, y_pred=y_preds)

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     #testing
#     model_3.eval()
#     with torch.inference_mode():
#         test_logits = model_3(x_test).squeeze()
#         test_preds = torch.round(torch.sigmoid(test_logits))

#         test_loss = loss_fn(test_logits, y_test)
#         test_acc = accuracy_fn(y_test=y_test, y_pred=test_preds)

#         # if epoch % 100 == 0:
#         #     print(f"Epoch: {epoch} | Train Loss: {loss:.2f}, Train Acc: {acc:.2f}% | Test Loss: {test_loss:.2f}, Test Acc: {test_acc:.2f}%")

# model_3.eval()
# with torch.inference_mode():
#     y_preds = torch.round(torch.sigmoid(model_3(x_test))).squeeze()

# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.title("Train")
# plot_decision_boundary(model_3, x_train, y_train)
# plt.subplot(1, 2, 2)
# plt.title("Test")
# plot_decision_boundary(model_3, x_test, y_test)
# plt.show()


#~~~~~~~~MULTI CLASSIFICATION~~~~~~~~
import torch
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch import nn
from helper_functions import plot_decision_boundary, plot_predictions

NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

x_blob, y_blob = make_blobs(n_samples=1000,
                            n_features=NUM_FEATURES,
                            centers=NUM_CLASSES,
                            cluster_std=1.5,
                            random_state=RANDOM_SEED)

x_blob = torch.from_numpy(x_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

x_train, x_test, y_train, y_test = train_test_split(x_blob, y_blob,
                                                    test_size=0.2,
                                                    random_state=RANDOM_SEED)

# plt.figure(figsize=(10, 7))
# plt.scatter(x_blob[:, 0], x_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)
# plt.show()

device = "cuda" if torch.cuda.is_available() else "cpu"

class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        super().__init__()
        # using sq stack this time
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units ),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_features),
        )

    def forward(self, x):
        return self.linear_layer_stack(x)
    
model_4 = BlobModel(input_features=NUM_FEATURES,
                    output_features=NUM_CLASSES,
                    hidden_units=8).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_4.parameters(), lr=0.1)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = ( correct / len(y_pred) ) * 100 #time 100 for percent
    return acc

# y_preds_prob = torch.softmax(model_4(x_test.to(device)))

torch.manual_seed(42)
epochs = 100

x_train = x_train.to(device)
y_train = y_train.to(device)
x_test = x_test.to(device)
y_test = y_test.to(device)

for epoch in range(epochs):
    model_4.train()
    y_logits = model_4(x_train)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

    loss = loss_fn(y_logits, y_train)
    # print(y_train.shape, y_pred.shape)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_4.eval()
    with torch.inference_mode():
        test_logits = model_4(x_test)
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)

        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.2f} | Acc: {acc:.2f}% | Test Loss: {test_loss:.2f} | Test Acc: {test_acc:.2f}%")


model_4.eval()
with torch.inference_mode():
    y_logits = model_4(x_test)
    y_preds = torch.softmax(y_logits, dim=1).argmax(dim=1)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_4, x_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_4, x_test, y_test)
plt.show()