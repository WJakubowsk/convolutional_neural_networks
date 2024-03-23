import sys

sys.path.append("..")
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from augmentation.augmentor import Augmentor


class ConvolutionalNeuralNetwork(torch.nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 10):
        """
        Convolution Neural Network model version with 3 convolutions.
        Args:
            in_channels: int, number of channels in the input image.
            out_channels: int, number of output classes.
        """
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels, 16, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(64 * 4 * 4, 128)
        self.fc2 = torch.nn.Linear(128, out_channels)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        Args:
            x: torch.Tensor, input tensor.
        Returns:
            x: torch.Tensor, output tensor.
        """
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)
        x = x.view(-1, 64 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

    def fit(
        self, x: torch.Tensor, y: torch.Tensor, epochs: int = 10, lr: float = 0.001
    ):
        """
        Fit the model to the data.
        Args:
            x: torch.Tensor, input tensor.
            y: torch.Tensor, target tensor.
            epochs: int, number of epochs.
            lr: float, learning rate.
        """
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.forward(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict the output of the model.
        Args:
            x: torch.Tensor, input tensor.
        Returns:
            torch.Tensor, predicted tensor.
        """
        with torch.no_grad():
            output = self.forward(x)
            return output.argmax(dim=1)

    def save(self, path: str):
        """
        Save the model to a file.
        Args:
            path: str, path where the model will be saved.
        """
        torch.save(self.state_dict(), path)


class ConvolutionalNeuralNetwork2(torch.nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 30):
        """
        Convolution Neural Network model version with 2 convolutions.
        Args:
            in_channels: int, number of channels in the input image.
            out_channels: int, number of output classes.
        """
        super(ConvolutionalNeuralNetwork2, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels, 16, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.fc1 = torch.nn.Linear(32 * 8 * 8, 128)
        self.fc2 = torch.nn.Linear(128, out_channels)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.avgpool = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        Args:
            x: torch.Tensor, input tensor.
        Returns:
            x: torch.Tensor, output tensor.
        """
        x = self.relu(self.conv1(x))
        x = self.avgpool(x)
        x = self.relu(self.conv2(x))
        x = self.avgpool(x)
        x = x.view(-1, 32 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

    def fit(
        self, x: torch.Tensor, y: torch.Tensor, epochs: int = 10, lr: float = 0.001
    ):
        """
        Fit the model to the data.
        Args:
            x: torch.Tensor, input tensor.
            y: torch.Tensor, target tensor.
            epochs: int, number of epochs.
            lr: float, learning rate.
        """
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.forward(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict the output of the model.
        Args:
            x: torch.Tensor, input tensor.
        Returns:
            torch.Tensor, predicted tensor.
        """
        with torch.no_grad():
            output = self.forward(x)
            return output.argmax(dim=1)

    def save(self, path: str):
        """
        Save the model to a file.
        Args:
            path: str, path where the model will be saved.
        """
        torch.save(self.state_dict(), path)


class ConvolutionalNeuralNetwork3(torch.nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 30):
        """
        Convolution Neural Network model with Leaky ReLU activation function.
        Args:
            in_channels: int, number of channels in the input image.
            out_channels: int, number of output classes.
        """
        super(ConvolutionalNeuralNetwork3, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels, 32, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.fc1 = torch.nn.Linear(128 * 8 * 8, 256)
        self.fc2 = torch.nn.Linear(256, out_channels)
        self.leakyrelu = torch.nn.LeakyReLU(negative_slope=0.01)
        self.softmax = torch.nn.Softmax(dim=1)
        self.avgpool = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        Args:
            x: torch.Tensor, input tensor.
        Returns:
            x: torch.Tensor, output tensor.
        """
        x = self.leakyrelu(self.bn1(self.conv1(x)))
        x = self.avgpool(x)
        x = self.leakyrelu(self.bn2(self.conv2(x)))
        x = self.avgpool(x)
        x = self.leakyrelu(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        x = x.view(-1, 128 * 8 * 8)
        x = self.leakyrelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

    def fit(
        self, x: torch.Tensor, y: torch.Tensor, epochs: int = 10, lr: float = 0.001
    ):
        """
        Fit the model to the data.
        Args:
            x: torch.Tensor, input tensor.
            y: torch.Tensor, target tensor.
            epochs: int, number of epochs.
            lr: float, learning rate.
        """
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.forward(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict the output of the model.
        Args:
            x: torch.Tensor, input tensor.
        Returns:
            torch.Tensor, predicted tensor.
        """
        with torch.no_grad():
            output = self.forward(x)
            return output.argmax(dim=1)

    def save(self, path: str):
        """
        Save the model to a file.
        Args:
            path: str, path where the model will be saved.
        """
        torch.save(self.state_dict(), path)


def main(args):
    # set seed
    seed = args.seed
    torch.manual_seed(seed)

    if args.model == "cnn":
        model = ConvolutionalNeuralNetwork()
    elif args.model == "cnn2":
        model = ConvolutionalNeuralNetwork2()
    elif args.model == "cnn3":
        model = ConvolutionalNeuralNetwork3()
    else:
        raise ValueError("Model not found.")

    # load data
    cinic_directory = args.cinic_directory  #'../../data/'
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]
    cinic_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            cinic_directory + "train",
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=cinic_mean, std=cinic_std),
                ]
            ),
        ),
        batch_size=128,
        shuffle=True,
    )
    cinic_test = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            cinic_directory + "test",
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=cinic_mean, std=cinic_std),
                ]
            ),
        ),
        batch_size=128,
    )

    # train model
    n_epochs = args.n_epochs
    augmentor = Augmentor()
    for epoch in range(1, n_epochs + 1):
        torch.manual_seed(epoch)

        print(f"EPOCH: {epoch}/{n_epochs}")

        train_loss = 0.0

        for batch_idx, sample in enumerate(cinic_train):
            inputs_list = augmentor.augment_data(sample["image"])
            labels = sample["label"]
            for input in inputs_list:
                model.fit(input, labels)
                loss = model.loss
                train_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Batch: {batch_idx}, Train Loss: {loss.item()}")
        print(f"Train Loss: {train_loss}")

        # evaluate model
        # accuracy
        correct = 0
        total = 0
        with torch.no_grad():
            for sample in cinic_test:
                inputs = sample["image"]
                labels = sample["label"]
                outputs = model.predict(inputs)
                total += labels.size(0)
                correct += (outputs == labels).sum().item()
        print(f"Accuracy: {100 * correct / total}")
        # confusion matrix
        y_true = []
        y_pred = []
        with torch.no_grad():
            for sample in cinic_test:
                inputs = sample["image"]
                labels = sample["label"]
                outputs = model.predict(inputs)
                y_true.extend(labels.numpy())
                y_pred.extend(outputs.numpy())
        confusion_matrix = pd.crosstab(
            pd.Series(y_true, name="Actual"),
            pd.Series(y_pred, name="Predicted"),
            margins=True,
        )
        print("confusion_matrix", confusion_matrix)
        # plot confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(confusion_matrix, annot=True)
        plt.savefig("confusion_matrix.png")
        # save model
        model.save(model + seed + "model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="cnn", help="Model type.")
    parser.add_argument(
        "--cinic_directory", type=str, default="../../data/", help="CINIC-10 directory."
    )
    parser.add_argument("--n_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    args = parser.parse_args()
    main(args)
