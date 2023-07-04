import torch.nn as nn

class InpaintingNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(InpaintingNetwork, self).__init__()

        # Calculate the intermediate size based on pooling
        intermediate_size = input_size // 8

        # Define the neural network layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #self.fc1 = nn.Linear(64 *  1024, intermediate_size * intermediate_size) # if input_size is 64x64
        self.fc1 = nn.Linear(256 * 1024, intermediate_size * intermediate_size)  # if input_size is 128x128
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(intermediate_size * intermediate_size, output_size * output_size * 3)
        self.output_size = output_size
        self.intermediate_size = intermediate_size

    def forward(self, x):
        # Implement the forward pass
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)

        # Flatten the tensor before feeding it to fully connected layers
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)

        # Reshape the output to have the shape (batch_size, 3, output_size, output_size)
        x = x.view(-1, 3, self.output_size, self.output_size)

        return x

