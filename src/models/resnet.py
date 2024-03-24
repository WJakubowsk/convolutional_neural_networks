import sys
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn.functional as F

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
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.forward(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            # print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

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
    n_classes = 10

    for epoch in tqdm(range(1, n_epochs + 1)):
        torch.manual_seed(epoch)

        print(f"EPOCH: {epoch}/{n_epochs}")

        # train model
        for batch_x, batch_y in tqdm(cinic_train):
            model.fit(batch_x, batch_y, epochs=1, lr=0.001)
            # print(batch_x.shape)
            # for i in range(len(batch_x)):
            #     augmented_images = augmentor.augment_data(batch_x[i])
            #     target_one_hot = F.one_hot(batch_y[i], n_classes).float()
            #     for j in range(len(augmented_images)):
            #         model.fit(augmented_images[j], target_one_hot, epochs=1, lr=0.001)

        # validate model
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in tqdm(cinic_valid):
                # for i in range(len(batch_x)):
                #     outputs = model.predict(batch_x[i])
                #     total += batch_y.size(0)
                #     correct += (outputs == batch_y[i].unsqueeze(0)).sum().item()
                outputs = model.predict(batch_x)
                total += batch_y.size(0)
                correct += (outputs == batch_y).sum().item()

        print(f"Accuracy on validation (%): {round(100 * correct / total, 2)}")

        with open("results/accuracy.txt", "a") as f:
            f.write(f"resnet,{seed},{epoch},{correct / total}")
    # test model
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch_x, batch_y in tqdm(cinic_test):
            # for i in range(len(batch_x)):
            #     outputs = model.predict(batch_x[i])
            #     y_true.extend(batch_y[i].unsqueeze(0).numpy())
            #     y_pred.extend(outputs.numpy())
            outputs = model.predict(batch_x)
            y_true.extend(batch_y.numpy())
            y_pred.extend(outputs.numpy())

    confusion_matrix = pd.crosstab(
        pd.Series(y_true, name="Actual"),
        pd.Series(y_pred, name="Predicted"),
        margins=True,
    )
    # plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix, annot=True)
    plt.savefig(f"results/resnet-{seed}-confusion_matrix.png")

    # save confusion matrix and accuracy to file
    confusion_matrix.to_csv(f"results/resnet-{seed}-confusion_matrix.csv")

    # save model
    model.save(f"pretrained/{args.model}-{seed}-model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cinic_directory", type=str, default="../../data/", help="CINIC-10 directory."
    )
    parser.add_argument("--n_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    args = parser.parse_args()
    main(args)
