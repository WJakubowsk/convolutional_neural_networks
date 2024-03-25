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
        self,
        network: str,
        classes_per_model: list[list[int]],
        lr: float = 0.001,
    ):
        """
        Initialize the ensemble model.
        Args:
            network: str, a convolutional neural network architecture, availble: cnn, cnn2, cnn3.
            classes_per_model: list[list[int]], the target classes for each model to fit on.
            lr: float, the learning rate.
        """
        self.models = []
        self.classes_per_model = classes_per_model
        self.num_models = len(classes_per_model)

        for num_classes in self.classes_per_model:
            # create model
            if network == "cnn":
                model = ConvolutionalNeuralNetwork(out_channels=len(num_classes))
            elif network == "cnn2":
                model = ConvolutionalNeuralNetwork2(out_channels=len(num_classes))
            elif network == "cnn3":
                model = ConvolutionalNeuralNetwork3(out_channels=len(num_classes))
            else:
                raise ValueError("Model not recognized.")
            self.models.append(model)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizers = [
            optim.Adam(model.parameters(), lr=lr) for model in self.models
        ]

    def fit(self, x: torch.Tensor, y: torch.Tensor, label_mapping: dict):
        """
        Fit the appropriate model to the training data.
        Args:
            x: torch.Tensor, the features.
            y: torch.Tensor, the target variable.
        """
        loss_total = 0
        for i, model in enumerate(self.models):
            # select observations and targets with targets only in the classes_per_model[i]
            mask = np.isin(y.numpy(), self.classes_per_model[i])
            if mask.any():
                x = x[mask]
                y = y[mask]
                mapped_labels = torch.tensor(
                    [
                        torch.tensor(label_mapping[label.item()], dtype=torch.long)
                        for label in y
                    ],
                    dtype=torch.long,
                )
                model.fit(x, mapped_labels)
                self.optimizers[i].zero_grad()
                outputs = model.forward(x)
                loss = self.criterion(outputs, mapped_labels)
                loss.backward()
                self.optimizers[i].step()
                loss_total += loss.item()
        return loss_total

    # def predict(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     Predict the output of the ensemble model by taking the maximum confidence of the models.
    #     """
    #     predictions = []
    #     for model in self.models:
    #         model.eval()
    #         with torch.no_grad():
    #             inputs = torch.tensor(x, dtype=torch.float32)
    #             outputs = model(inputs)
    #             predictions.append(outputs.numpy())
    #     max_confidence = np.max(predictions, axis=0)
    #     ensemble_predictions = np.argmax(max_confidence, axis=1)
    #     return ensemble_predictions

    def predict(self, x, label_mapping: dict):
        predictions = [[] for _ in range(self.num_models)]
        model_indices = [[] for _ in range(self.num_models)]
        for idx, model in enumerate(self.models):
            model.eval()
            # print(x)
            with torch.no_grad():
                # inputs = torch.tensor(x, dtype=torch.float32)
                outputs = model(x)
                # print(outputs.numpy())
                model_predictions = np.max(outputs.numpy(), axis=1)  # .tolist()
                model_predictions_index = np.argmax(outputs.numpy(), axis=1)
                predictions[idx].extend(model_predictions)
                model_indices[idx].extend(model_predictions_index)

        # Find the maximum prediction for each observation
        # print(predictions[0])

        max_elements = []
        max_indices = []

        for values in zip(*predictions):
            max_val = max(values)
            max_elements.append(max_val)
            max_idx = values.index(max_val)
            max_indices.append(max_idx)

        # print(model_indices, len(model_indices[0]))
        # print(max_indices, len(max_indices))

        # Return mapped predictions
        mapped_predictions = [
            label_mapping[el][model_indices[el][i]] for i, el in enumerate(max_indices)
        ]
        # print(mapped_predictions, len(mapped_predictions))
        return torch.tensor(mapped_predictions, dtype=torch.long)


def main(args):
    # set seed
    seed = args.seed
    torch.manual_seed(seed)

    ensemble = EnsembleCNN(network=args.model, classes_per_model=args.classes_per_model)

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
    label_mapping = {0: 0, 1: 0, 2: 1, 3: 0, 4: 0, 5: 1, 6: 2, 7: 1, 8: 3, 9: 1}
    reversed_mapping = {
        0: {1: 9, 0: 1},
        1: {1: 5, 0: 3},
        2: {1: 7, 0: 4},
        3: {3: 8, 2: 6, 1: 2, 0: 0},
    }

    for epoch in range(1, n_epochs + 1):
        # train model
        loss = 0
        for batch_x, batch_y in cinic_train:
            loss += ensemble.fit(batch_x, batch_y, label_mapping)

        print(f"Epoch {epoch}, Loss: {loss / len(cinic_train.dataset)}")

        # validate model
        correct_train = 0
        correct_valid = 0
        total_train = 0
        total_valid = 0

        with torch.no_grad():
            for batch_x, batch_y in cinic_train:
                outputs = ensemble.predict(batch_x, reversed_mapping)
                total_train += batch_y.size(0)
                correct_train += (outputs == batch_y).sum().item()

            for batch_x, batch_y in cinic_valid:
                outputs = ensemble.predict(batch_x, reversed_mapping)
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
            outputs = ensemble.predict(batch_x, reversed_mapping)
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
    plt.savefig(
        f"{args.outputdir}/results/ensemble-{args.model}-{seed}-confusion_matrix.png"
    )

    # save confusion matrix and accuracy to file
    confusion_matrix.to_csv(
        f"{args.outputdir}/results/ensemble-{args.model}-{seed}-confusion_matrix.csv"
    )

    # save model
    model.save(f"{args.outputdir}/pretrained/ensemble-{args.model}-{seed}-model.pth")


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
        "--classes_per_model",
        type=list,
        default=[[1, 9], [3, 5], [4, 7], [0, 2, 6, 8]],
        help="classes per network.",
    )
    args = parser.parse_args()
    main(args)
