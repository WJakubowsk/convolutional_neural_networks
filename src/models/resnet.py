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


class ResNet50(nn.Module):
    def __init__(self, num_classes=10, lr=0.001):
        super(ResNet50, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def fit(self, x: torch.Tensor, y: torch.Tensor):
        """
        Fit the model to the data on one epoch.
        """
        self.optimizer.zero_grad()
        output = self.forward(x)
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict the output of the model.
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
    cinic_directory = args.data
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
    n_epochs = args.epochs

    for epoch in tqdm(range(1, n_epochs + 1)):
        torch.manual_seed(seed + epoch)

        for batch_x, batch_y in cinic_train:
            output = model.fit(batch_x, batch_y)

        print(f"Epoch {epoch}, Loss train: {output}")

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

        with open(f"{args.outputdir}/results/accuracy.txt", "a") as f:
            f.write(f"resnet,{seed},{epoch},{correct / total}")
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
    plt.savefig(f"{args.outputdir}/results/resnet-{seed}-confusion_matrix.png")

    # save confusion matrix and accuracy to file
    confusion_matrix.to_csv(f"{args.outputdir}/results/resnet-{seed}-confusion_matrix.csv")

    # save model
    model.save(f"{args.outputdir}/pretrained/resnet-{seed}-model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, default="../../data/", help="CINIC-10 directory."
    )
    parser.add_argument("--outputdir", type=str, default="../../src/models", help="CINIC-10 directory."
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    args = parser.parse_args()
    main(args)
