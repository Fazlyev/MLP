import torch
import numpy as np
import pandas as pd
from torch import nn
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tqdm.notebook import tqdm


class MLP(nn.Module):
  def __init__(self, in_features, num_classes, hidden_size):
    super().__init__()
    self.model = nn.Sequential(
        nn.Linear(in_features=in_features, out_features=hidden_size), # y1 = f1(x, w1) 64x784
        nn.ReLU(), # y2 = ReLu(y1, w2) 64
        nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True), # y3 = f3(y2, w3) 64x64
        nn.LeakyReLU(0.1), # y4 = LeakyReLu(y3, w4) 64
        nn.Linear(in_features=hidden_size, out_features=num_classes) # y5 = f5(y4, w5) 10x64
    )
  def forward(self, x):
    return self.model(x)


def plot_losses(train_losses, test_losses, train_accuracies, test_accuracies):
    clear_output()
    sns.set_style('whitegrid')
    plt.rcParams.update({'font.size': 15})
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    axs[0].plot(range(1, len(train_losses) + 1), train_losses, label='train')
    axs[0].plot(range(1, len(test_losses) + 1), test_losses, label='test')
    axs[0].set_ylabel('loss')

    axs[1].plot(range(1, len(train_accuracies) + 1), train_accuracies, label='train')
    axs[1].plot(range(1, len(test_accuracies) + 1), test_accuracies, label='test')
    axs[1].set_ylabel('accuracy')

    for ax in axs:
        ax.set_xlabel('epoch')
        ax.legend()

    plt.show()


def train(model=None, device=None, NUM_EPOCHS=None, optimizer=None, criterion=None, train_loader=None, test_loader=None):
  train_losses, train_accuracies = [], []
  test_losses, test_accuracies = [], []

  for epoch in range(1, NUM_EPOCHS+1):
    train_loss, train_accuracy = 0.0, 0.0
    model.train()

    for images, labels in tqdm(train_loader, desc='Training'): # берем объекты с их истинными метками
      images = images.to(device)
      labels = labels.to(device) # кладем куда надо

      optimizer.zero_grad() # зануляем градиенты с прошлого объекта
      logits = model(torch.flatten(images, start_dim=1)) # логиты, полученные при проходе вперед
      loss = criterion(logits, labels) # кросс-энтропия на этих логитах
      loss.backward() # считаем градиент ошибки
      optimizer.step() # делаем шаг SGD в соответствии с этим градиентом

      train_loss += loss.item()*images.shape[0] # умножаем на число объектов в подвыборке
      train_accuracy += (logits.argmax(dim=1)==labels).sum().item() # каждый логит это 10 вероятностей для каждого объекта, выбираем для каждого самый вероятный класс и сравниваем с истинным, считаем сумму

    train_loss/=len(train_loader.dataset)
    train_accuracy/=len(train_loader.dataset)
    train_losses+=[train_loss]
    train_accuracies+=[train_accuracy]

    test_loss, test_accuracy = 0.0, 0.0
    model.eval()

    for images, labels in tqdm(test_loader, desc='Validating'):
      images = images.to(device)
      labels = labels.to(device) # кладем куда надо

      with torch.no_grad():
        logits = model(torch.flatten(images, start_dim=1))
        loss = criterion(logits, labels)
      test_loss+=loss.item()*images.shape[0]
      test_accuracy+=(logits.argmax(dim=1)==labels).sum().item()

    test_loss/=len(test_loader.dataset)
    test_accuracy/=len(test_loader.dataset)
    test_losses+=[test_loss]
    test_accuracies+=[test_accuracy]

    plot_losses(train_losses, test_losses, train_accuracies, test_accuracies)