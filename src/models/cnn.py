import sys
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

sys.path.append("..")
from augmentation.augmentor import Augmentor


class ConvolutionalNeuralNetwork(torch.nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 10, lr: float = 0.01):
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
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

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

    def fit(self, x: torch.Tensor, y: torch.Tensor, epochs: int = 10):
        """
        Fit the model to the data.
        Args:
            x: torch.Tensor, input tensor.
            y: torch.Tensor, target tensor.
            epochs: int, number of epochs.
            lr: float, learning rate.
        """
        for _ in range(epochs):
            self.optimizer.zero_grad()
            output = self.forward(x)
            loss = self.criterion(output, y)
            loss.backward()
            self.optimizer.step()

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
    def __init__(self, in_channels: int = 3, out_channels: int = 10, lr: float = 0.01):
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
        self.fc1 = torch.nn.Linear(32 * 4 * 4, 128)
        self.fc2 = torch.nn.Linear(128, out_channels)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.avgpool = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout = torch.nn.Dropout(0.5)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

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
        x = x.view(-1, 32 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

    def fit(self, x: torch.Tensor, y: torch.Tensor, epochs: int = 10):
        """
        Fit the model to the data.
        Args:
            x: torch.Tensor, input tensor.
            y: torch.Tensor, target tensor.
            epochs: int, number of epochs.
            lr: float, learning rate.
        """
        for _ in range(epochs):
            self.optimizer.zero_grad()
            output = self.forward(x)
            loss = self.criterion(output, y)
            loss.backward()
            self.optimizer.step()

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
    def __init__(self, in_channels: int = 3, out_channels: int = 10, lr: float = 0.001):
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
        self.fc1 = torch.nn.Linear(128 * 4 * 4, 256)
        self.fc2 = torch.nn.Linear(256, out_channels)
        self.leakyrelu = torch.nn.LeakyReLU(negative_slope=0.01)
        self.softmax = torch.nn.Softmax(dim=1)
        self.avgpool = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout = torch.nn.Dropout(0.5)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        Args:
            x: torch.Tensor, input tensor.
        Returns:
            x: torch.Tensor, output tensor.
        """
        x = x.unsqueeze(0)
        x = self.leakyrelu(self.bn1(self.conv1(x)))
        x = self.avgpool(x)
        x = self.leakyrelu(self.bn2(self.conv2(x)))
        x = self.avgpool(x)
        x = self.leakyrelu(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.leakyrelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def fit(self, x: torch.Tensor, y: torch.Tensor):
        """
        Fit the model to the data on one epoch.
        Args:
            x: torch.Tensor, input tensor.
            y: torch.Tensor, target tensor.
            epochs: int, number of epochs.
            lr: float, learning rate.
        """
        self.optimizer.zero_grad()
        output = self.forward(x)
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()

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
    n_classes = 10

    if args.model == "cnn":
        model = ConvolutionalNeuralNetwork(out_channels=n_classes)
    elif args.model == "cnn2":
        model = ConvolutionalNeuralNetwork2(out_channels=n_classes)
    elif args.model == "cnn3":
        model = ConvolutionalNeuralNetwork3(out_channels=n_classes)
    else:
        raise ValueError("Model not found.")

    # load data
    cinic_directory = args.cinic_directory
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]
    cinic_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            cinic_directory + "train",
            transform=transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
                    transforms.RandomRotation(
                        degrees=30
                    ),  # Randomly rotate the image by up to 30 degrees
                    transforms.RandomApply(
                        [
                            transforms.Lambda(
                                lambda img: transforms.functional.adjust_sharpness(
                                    img, sharpness_factor=2.0
                                )
                            ),  # Increase edge sharpness
                        ],
                        p=0.25,
                    ),  # random sharpness
                    transforms.RandomApply(
                        [
                            transforms.GaussianBlur(
                                kernel_size=3
                            ),  # Apply Gaussian blur
                        ],
                        p=0.25,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=cinic_mean, std=cinic_std),
                ]
            ),
        ),
        batch_size=128,
        shuffle=True,
    )

    cinic_valid = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            cinic_directory + "valid",
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
        shuffle=True,
    )

    n_epochs = args.epochs

    for epoch in tqdm(range(1, n_epochs + 1)):
        torch.manual_seed(seed + epoch)

        # train model
        for batch_x, batch_y in tqdm(cinic_train):
            model.fit(batch_x, batch_y)

        # validate model
        correct_train = 0
        correct_valid = 0
        total_train = 0
        total_valid = 0

        with torch.no_grad():
            for batch_x, batch_y in cinic_train:
                outputs = model.predict(batch_x)
                total_train += batch_y.size(0)
                correct_train += (outputs == batch_y).sum().item()

            for batch_x, batch_y in cinic_valid:
                outputs = model.predict(batch_x)
                total_valid += batch_y.size(0)
                correct_valid += (outputs == batch_y).sum().item()
        print(f"Accuracy on train (%): {round(100 * correct_train / total_train, 2)}")
        print(
            f"Accuracy on validation (%): {round(100 * correct_valid / total_valid, 2)}"
        )

        with open("results/accuracy.txt", "a") as f:
            f.write(
                f"{args.model},{seed},{epoch},{correct_train / total_train},{correct_valid / total_valid}\n"
            )

    # test model
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch_x, batch_y in cinic_test:
            outputs = model.predict(batch_x)
            y_true.extend(batch_y.numpy())
            y_pred.extend(outputs.numpy())

    confusion_matrix = pd.crosstab(
        pd.Series(y_true, name="Actual"),
        pd.Series(y_pred, name="Predicted"),
        margins=False,
    )
    # plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix, annot=True)
    plt.savefig(f"results/{args.model}-{seed}-confusion_matrix.png")

    # save confusion matrix and accuracy to file
    confusion_matrix.to_csv(f"results/{args.model}-{seed}-confusion_matrix.csv")

    # save model
    model.save(f"pretrained/{args.model}-{seed}-model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="cnn", help="Model type.")
    parser.add_argument(
        "--cinic_directory", type=str, default="../../data/", help="CINIC-10 directory."
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    args = parser.parse_args()
    main(args)
