import torch


class LeNet5(torch.nn.Module):
    """
    The LeNet-5 module.
    """

    def __init__(self):

        # Mandatory call to super class module.
        super(LeNet5, self).__init__()

        # Defining the feature extraction layers.
        self.feature_extractor = torch.nn.Sequential(

            # Layer 1 - Conv2d(4, 5x5) - Nx1x32x32 -> Nx6x28x28
            torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            torch.nn.Tanh(),

            # Layer 2 - AvgPool2d(2x2) - Nx6x28x28 -> Nx6x14x14
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
            torch.nn.Sigmoid(),

            # Layer 3 - Conv2d(12, 5x5) - Nx6x14x14 -> Nx16x10x10
            torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            torch.nn.Tanh(),

            # Layer 4 - AvgPool2d(2x2) - Nx16x10x10 -> Nx16x5x5
            self.s4=torch.nn.AvgPool2d(kernel_size=2, stride=2),
            torch.nn.Sigmoid(),
        )

        # Defining the classification layers.
        self.classifier = torch.nn.Sequential(

            # Layer 5 - FullyConnected(120) - Nx1x400 -> Nx1x120
            torch.nn.Linear(in_features=16*5*5, out_features=120),
            torch.nn.Tanh(),

            # Layer 6 - FullyConnected(10) - Nx1x120 -> Nx1x84
            torch.nn.Linear(in_features=120, out_features=84),
            torch.nn.Tanh(),

            # Layer 7 - FullyConnected(10) - Nx1x84 -> Nx1x10
            torch.nn.Linear(in_features=84, out_features=10),
            torch.nn.Softmax()
        )

    def forward(self, x):

        # Forward pass through the feature extractor - Nx1x32x32 -> Nx16x5x5
        x = self.feature_extractor(x)

        # Flattening the feature map - Nx16x5x5 -> Nx1x400
        x = torch.flatten(x, 1)

        # Forward pass through the classifier 5 to 7 - Nx1x400 -> Nx1x10
        return self.classifier(x)
