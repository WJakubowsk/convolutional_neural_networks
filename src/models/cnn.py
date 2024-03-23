import torch


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
        self.conv1 = torch.nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
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
    
    def fit(self, x: torch.Tensor, y: torch.Tensor, epochs: int = 10, lr: float = 0.001):
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

