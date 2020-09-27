import torch


class LeNet1(torch.nn.Module):
    """
    The LeNet-1 module.
    """

    def __init__(self):

        # Mandatory call to super class module.
        super(LeNet1, self).__init__()

        # Defining the feature extraction layers.
        self.feature_extractor = torch.nn.Sequential(

            # Layer 1 - Conv2d(4, 5x5) - Nx1x28x28 -> Nx4x24x24
            torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5),
            torch.nn.Tanh(),

            # Layer 2 - AvgPool2d(2x2) - Nx4x24x24 -> Nx4x12x12
            torch.nn.AvgPool2d(kernel_size=2, stride=2),

            # Layer 3 - Conv2d(12, 5x5) - Nx4x12x12 -> Nx12x8x8
            torch.nn.Conv2d(in_channels=4, out_channels=12, kernel_size=5),
            torch.nn.Tanh(),

            # Layer 4 - AvgPool2d(2x2) - Nx12x8x8 -> Nx12x4x4
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
        )

        # Defining the classification layers.
        self.classifier = torch.nn.Sequential(

            # Layer 5 - FullyConnected(10) - Nx1x192 -> Nx1x10
            torch.nn.Linear(in_features=12*4*4, out_features=10),
            torch.nn.Softmax()
        )

    def forward(self, x):

        # Forward pass through the feature extractor - Nx1x28x28 -> Nx12x4x4
        x = self.feature_extractor(x)

        # Flattening the feature map - Nx12x4x4 -> Nx1x192
        x = torch.flatten(x, 1)

        # Forward pass through the classifier - Nx1x192 -> Nx1x10
        return self.classifier(x)
