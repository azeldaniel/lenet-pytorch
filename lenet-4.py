import torch


class LeNet4(torch.nn.Module):
    """
    The LeNet-4 module.
    """

    def __init__(self):

        # Mandatory call to super class module.
        super(LeNet4, self).__init__()

        # Defining the feature extraction layers.
        self.feature_extractor = torch.nn.Sequential(

            # Layer 1 - Conv2d(4, 5x5) - Nx1x32x32 -> Nx4x28x28
            torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5),
            torch.nn.Tanh(),

            # Layer 2 - AvgPool2d(2x2) - Nx4x28x28 -> Nx4x14x14
            torch.nn.AvgPool2d(kernel_size=2, stride=2),

            # Layer 3 - Conv2d(12, 5x5) - Nx4x14x14 -> Nx16x10x10
            torch.nn.Conv2d(in_channels=4, out_channels=16, kernel_size=5),
            torch.nn.Tanh(),

            # Layer 4 - AvgPool2d(2x2) - Nx16x10x10 -> Nx16x5x5
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
        )

        # Defining the classification layers.
        self.classifier = torch.nn.Sequential(

            # Layer 5 - FullyConnected(120) - Nx1x400 -> Nx1x120
            torch.nn.Linear(in_features=16*5*5, out_features=120),
            torch.nn.Tanh(),

            # Layer 6 - FullyConnected(10) - Nx1x120 -> Nx1x10
            self.f6=torch.nn.Linear(in_features=120, out_features=10),
            torch.nn.Softmax(),
        )

    def forward(self, x):

        # Forward pass through the feature extractor - Nx1x32x32 -> Nx16x5x5
        x = self.feature_extractor(x)

        # Flattening the feature map - Nx16x5x5 -> Nx1x400
        x = torch.flatten(x, 1)

        # Forward pass through the classifier - Nx1x400 -> Nx1x10
        return self.classifier(x)
