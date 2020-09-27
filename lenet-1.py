import torch


class LeNet1(torch.nn.Module):
    """
    The LeNet-1 module.
    """

    def __init__(self):

        # Mandatory call to super class module.
        super(LeNet1, self).__init__()

        # Layer 1 - Conv2d(4, 5x5) - Nx1x28x28 -> Nx4x24x24
        self.c1 = torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5)

        # Layer 2 - AvgPool2d(2x2) - Nx4x24x24 -> Nx4x12x12
        self.s2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        # Layer 3 - Conv2d(12, 5x5) - Nx4x12x12 -> Nx12x8x8
        self.c3 = torch.nn.Conv2d(
            in_channels=4, out_channels=12, kernel_size=5)

        # Layer 4 - AvgPool2d(2x2) - Nx12x8x8 -> Nx12x4x4
        self.s4 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        # Layer 5 - FullyConnected(10) - Nx12x4x4 -> Nx1x10
        self.f5 = torch.nn.Linear(in_features=12*4*4, out_features=10)

    def forward(self, x):

        # Forward pass through layer 1, and tanh activation
        x = torch.tanh(self.c1(x))

        # Forward pass through layer 2
        x = self.s2(x)

        # Forward pass through layer 3, and tanh activation
        x = torch.tanh(self.c3(x))

        # Forward pass through layer 4
        x = self.s4(x)
        x = torch.flatten(x, 1)

        # Forward pass through layer 5, and softmax activation
        return torch.nn.functional.softmax(self.f5(x))

    def flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
