import torch


class LeNet5(torch.nn.Module):
    """
    The LeNet-5 module.
    """

    def __init__(self):

        # Mandatory call to super class module.
        super(LeNet5, self).__init__()

        # Layer 1 - Conv2d(4, 5x5) - Nx1x32x32 -> Nx6x28x28
        self.c1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)

        # Layer 2 - AvgPool2d(2x2) - Nx6x28x28 -> Nx6x14x14
        self.s2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        # Layer 3 - Conv2d(12, 5x5) - Nx6x14x14 -> Nx16x10x10
        self.c3 = torch.nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5)

        # Layer 4 - AvgPool2d(2x2) - Nx16x10x10 -> Nx16x5x5
        self.s4 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        # Layer 5 - FullyConnected(120) - Nx16x5x5 -> Nx1x120
        self.f5 = torch.nn.Linear(in_features=16*5*5, out_features=120)

        # Layer 6 - FullyConnected(10) - Nx1x120 -> Nx1x84
        self.f6 = torch.nn.Linear(in_features=120, out_features=84)

        # Layer 7 - FullyConnected(10) - Nx1x84 -> Nx1x10
        self.f7 = torch.nn.Linear(in_features=84, out_features=10)

    def forward(self, x):

        # Forward pass through layer 1, and tanh activation
        x = torch.tanh(self.c1(x))

        # Forward pass through layer 2, and sigmoid activation
        x = torch.sigmoid(self.s2(x))

        # Forward pass through layer 3, and tanh activation
        x = torch.tanh(self.c3(x))

        # Forward pass through layer 4,  and sigmoid activation
        x = torch.sigmoid(self.s4(x))
        x = torch.flatten(x, 1)

        # Forward pass through layer 5, and tanh activation
        x =  torch.tanh(self.f5(x))

        # Forward pass through layer 6, and tanh activation
        x =  torch.tanh(self.f6(x))

        # Forward pass through layer 7, and softmax activation
        return torch.nn.functional.softmax(self.f7(x))
