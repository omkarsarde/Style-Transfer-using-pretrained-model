
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import SGD, Adam
import torchvision
import tarfile
import re
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.preprocessing import LabelEncoder

"""
Using Pretrained Networks to transfer features, effectively acting as Feature Extractors
to reduce training times
"""

class Data(Dataset):
    """
    Fundamental implementation of Dataset class of pyTorch
    """
    def __init__(self, root_dir, transform=None):
        """
        Intialization
        """
        self.root_dir = root_dir
        self.transform = transform
        self.labels, self.samples = [], []
        self._init_dataset()
        self.labels = LabelEncoder().fit_transform(self.labels)

    def __len__(self):
        """
        Get the length of dataset
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset using id
        """
        x = self.samples[idx]
        if self.transform:
            x = self.transform(x)
        y = self.labels[idx]
        return x, y

    def _init_dataset(self):
        """
        Read folder for dataset generation
        """
        for file in glob.iglob(self.root_dir + "/*.jpg"):
            name = file.split("\\")[-1]
            self.labels.append(re.split(r'(\d+)', name, maxsplit=1)[0])
            img = Image.open(file)
            img = img.convert('RGB')
            self.samples.append(img)


def imshow(img, title=''):
    """Plot the image batch.
    """
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)), cmap='gray')
    plt.show()


def main():
    """
    Driver function
    """
    device = torch.device("cuda")
    resize = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = Data('./Dataset/images', transform=resize)
    train_set, test_set = random_split(dataset, [5173, 2217])
    train_loader = DataLoader(train_set, batch_size=200, shuffle=True, num_workers=6)
    test_loader = DataLoader(test_set, batch_size=200, shuffle=True, num_workers=4)

    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.classifier[-1] = nn.Sequential(nn.Linear(in_features=4096, out_features=37),
                                         nn.LogSoftmax(dim=1))
    model.to(device)
    print(model)
    optimizer = Adam(model.parameters())
    loss_criterion = nn.NLLLoss()
    epochs = 10
    batch_loss = 0
    total_epoch_loss = 0
    history = []
    # Fundamental Torch training loop
    for epoch in range(epochs):
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        model.train()

        train_loss = 0.0
        train_acc = 0.0

        valid_loss = 0.0
        valid_acc = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            train_acc += acc.item() * inputs.size(0)

            print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))

        with torch.no_grad():

            model.eval()

            for j, (inputs, labels) in enumerate(test_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                loss = loss_criterion(outputs, labels)

                valid_loss += loss.item() * inputs.size(0)

                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                valid_acc += acc.item() * inputs.size(0)

                print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j,
                                                                                                           loss.item(),
                                                                                                           acc.item()))

        avg_train_loss = train_loss / 5173
        avg_train_acc = train_acc / float(5173)

        avg_valid_loss = valid_loss / 2217
        avg_valid_acc = valid_acc / float(2217)

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        print(
            "Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation : Loss : {:.4f}, Accuracy: {:.4f}%".format(
                epoch, avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100))

    history = np.array(history)
    print("HISTORY", history)
    plt.plot(history[:, 0:2])
    plt.legend(['Tr Loss', 'Val Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.ylim(0, 1)
    plt.savefig('_loss_curve.png')
    plt.show()

    plt.plot(history[:, 2:4])
    plt.legend(['Tr Accuracy', 'Val Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig('_accuracy_curve.png')
    plt.show()


if __name__ == '__main__':
    main()
