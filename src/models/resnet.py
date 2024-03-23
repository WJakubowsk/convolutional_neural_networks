import sys
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

sys.path.append("..")
from augmentation.augmentor import Augmentor


class ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
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
        criterion = nn.CrossEntropyLoss()
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


def main(args):
    # set seed
    seed = args.seed
    torch.manual_seed(seed)

    # create model
    model = ResNet50()

    # load data
    cinic_directory = args.cinic_directory
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

        for batch_x, batch_y in cinic_train:
            for i in range(len(batch_x)):
                batch_x[i] = augmentor.augment_data(batch_x[i])
                model.fit(batch_x[i], batch_y[i], epochs=1, lr=0.001)

        # evaluate model
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in cinic_valid:
                for i in range(len(batch_x)):
                    outputs = model.predict(batch_x[i])
                    total += batch_y[i].size(0)
                    correct += (outputs == batch_y[i]).sum().item()

        print(f"Accuracy: {100 * correct / total}")

        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch_x, batch_y in cinic_test:
                for i in range(len(batch_x)):
                    outputs = model.predict(batch_x[i])
                    y_true.extend(batch_y[i].numpy())
                    y_pred.extend(outputs.numpy())
        confusion_matrix = pd.crosstab(
            pd.Series(y_true, name="Actual"),
            pd.Series(y_pred, name="Predicted"),
            margins=True,
        )
        # plot confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(confusion_matrix, annot=True)
        plt.savefig("confusion_matrix.png")

        # save confusion matrix and accuracy to file
        confusion_matrix.to_csv(model + seed + "confusion_matrix.csv")
        with open("accuracy.txt", "w") as f:
            f.write(f"{model},{seed},{100 * correct / total}")

        # save model
        model.save(model + seed + "model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cinic_directory", type=str, default="../../data/", help="CINIC-10 directory."
    )
    parser.add_argument("--n_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    args = parser.parse_args()
    main(args)
