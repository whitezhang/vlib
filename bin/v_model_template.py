# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np
import matplotlib.pyplot as plt
import io
import sys
import os
from datetime import datetime


class ModelInterface:
    """
    ModelInterface
    """

    def __init__(self, model, epoch=10, batch_size=64, sw_name="interface_test"):
        """
        __init__
        """
        self.epoch = epoch
        self.batch_size = batch_size
        self.accuracy = 0.0

        self.device = torch.device("mps")
        self.index = 0

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            model.fc.parameters(), lr=1e-3, weight_decay=1e-3, momentum=0.9
        )

        self.model = model.to(self.device)
        self.writer = SummaryWriter(sw_name)

    def load_data_template(self):
        """
        load_data_template
        """
        transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

        train_data = datasets.MNIST(
            root=os.getcwd(), train=False, transform=transform, download=False
        )
        test_data = datasets.MNIST(
            root=os.getcwd(), train=False, transform=transform, download=False
        )

        self.train_loader = DataLoader(
            dataset=train_data, batch_size=self.batch_size, shuffle=True
        )
        self.test_loader = DataLoader(
            dataset=test_data, batch_size=self.batch_size, shuffle=True
        )

    def eval_template(self):
        """
        eval_template
        """
        correct, total = 0, 0
        for i, data in enumerate(self.test_loader):
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total = total + labels.size(0)
            correct = correct + (predicted == labels).sum().item()
        accuracy = 100.0 * correct / total
        self.writer.add_scalar("Train/Accuracy", accuracy, self.index)
        print("Accuracy：{:.4f}%({}/{})".format(accuracy, correct, total))

        self.accuracy = accuracy

    def train_template(self):
        """
        train_template
        """
        for self.index in range(1, self.epoch + 1):
            st = datetime.now()
            for i, data in enumerate(self.train_loader):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()

            et = datetime.now()
            self.eval_template()

            writer.add_scalar("Train/Loss", loss.item(), self.index)
            print(
                "epoch{} loss:{:.4f} cost:{}".format(
                    self.index, loss.item(), (et - st).seconds
                )
            )

    def load_model_template(self, fname):
        """
        load_model_template
        """
        self.model = torch.load(fname, map_location=torch.device("mps"))

    def save_model_template(self, fname=""):
        """
        save_model_template
        """
        if fname == "":
            fname = "./models/e-{}_ac-{:.2f}.pt" % (self.index, self.accuracy)
        torch.save(self.model, fname)
