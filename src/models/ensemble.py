import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from cnn import (
    ConvolutionalNeuralNetwork,
    ConvolutionalNeuralNetwork2,
    ConvolutionalNeuralNetwork3,
)


class EnsembleCNN:
    def __init__(
        self, network: nn.Module, num_classes_per_model: list[int], lr: float = 0.001
    ):
        """
        Initialize the ensemble model.
        Args:
            network: nn.Module, a convolutional neural network architecture, availble in cnn.py.
            num_classes_per_model: list[int], the number of classes for each model.
            lr: float, the learning rate.
        """
        self.models = []
        self.num_classes_per_model = num_classes_per_model
        self.num_models = len(num_classes_per_model)

        for num_classes in num_classes_per_model:
            model = network(num_classes)
            self.models.append(model)

        self.loss = nn.CrossEntropyLoss()
        self.optimizers = [
            optim.Adam(model.parameters(), lr=lr) for model in self.models
        ]

    def fit(self, x: torch.Tensor, y: torch.Tensor, model_index: int):
        """
        Fit the appropriate model to the training data.
        Args:
            x: torch.Tensor, the features.
            y: torch.Tensor, the target variable.
            model_index: int, the index of the model.
        """
        model = self.models[model_index]
        indices = np.where(
            np.isin(y, np.arange(sum(self.num_classes_per_model[model_index:])))
        )[0]
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        for i in range(0, len(indices), 32):
            inputs = torch.tensor(x[indices[i : i + 32]], dtype=torch.float32)
            targets = torch.tensor(y[indices[i : i + 32]], dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict the output of the ensemble model by taking the maximum confidence of the models.
        """
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                inputs = torch.tensor(x, dtype=torch.float32)
                outputs = model(inputs)
                predictions.append(outputs.numpy())
        max_confidence = np.max(predictions, axis=0)
        ensemble_predictions = np.argmax(max_confidence, axis=1)
        return ensemble_predictions


def main(args):
    # set seed
    seed = args.seed
    torch.manual_seed(seed)

    # create model
    if args.model == "cnn":
        model = ConvolutionalNeuralNetwork()
    elif args.model == "cnn2":
        model = ConvolutionalNeuralNetwork2()
    elif args.model == "cnn3":
        model = ConvolutionalNeuralNetwork3()
    else:
        raise ValueError("Model not recognized.")

    ensemble = EnsembleCNN(model, args.num_classes_per_model)

    # load data
    cinic_directory = args.data
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]
    cinic_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            cinic_directory + "train",
            transform=transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(degrees=30),
                    transforms.RandomApply(
                        [
                            transforms.Lambda(
                                lambda img: transforms.functional.adjust_sharpness(
                                    img, sharpness_factor=2.0
                                )
                            ),
                        ],
                        p=0.25,
                    ),
                    transforms.RandomApply(
                        [
                            transforms.GaussianBlur(kernel_size=3),
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

    n_epochs = args.epochs

    for epoch in range(1, n_epochs + 1):
        # train model
        for batch_x, batch_y in cinic_train:
            for j in range(ensemble.num_models):
                ensemble.fit(batch_x, batch_y, j)

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
            f.write(
                f"ensemble-{args.model},{seed},{epoch},{correct_train / total_train},{correct_valid / total_valid}\n"
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
    plt.savefig(f"{args.outputdir}/results/{args.model}-{seed}-confusion_matrix.png")

    # save confusion matrix and accuracy to file
    confusion_matrix.to_csv(
        f"{args.outputdir}/results/{args.model}-{seed}-confusion_matrix.csv"
    )

    # save model
    model.save(f"{args.outputdir}/pretrained/{args.model}-{seed}-model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="cnn3", help="Model type to create ensembles from."
    )
    parser.add_argument(
        "--data", type=str, default="../../data/", help="CINIC-10 directory."
    )
    parser.add_argument(
        "--outputdir", type=str, default="../../src/models", help="CINIC-10 directory."
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    parser.add_argument(
        "--num_classes_per_model",
        type=list,
        default=[3, 4, 3],
        help="Number of classes per network.",
    )
    args = parser.parse_args()
    main(args)
