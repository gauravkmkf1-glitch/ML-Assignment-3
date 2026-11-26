import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim=3072, hidden=(512, 256), classes=10):
        super().__init__()
        layers = []
        width = in_dim
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU()]
            width = h
        layers.append(nn.Linear(width, classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.flatten(1))


class CNN(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Linear(4096, 384), nn.ReLU(), nn.Linear(384, classes))

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))

    def penultimate(self, x):
        return self.classifier[:2](self.features(x).flatten(1))


class Autoencoder(nn.Module):
    def __init__(self, bottleneck=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.to_code = nn.Linear(4096, bottleneck)
        self.from_code = nn.Linear(bottleneck, 4096)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(32, 3, 3, padding=1), nn.Sigmoid(),
        )

    def encode(self, x):
        return self.to_code(self.encoder(x).flatten(1))

    def forward(self, x):
        z = self.encode(x)
        h = self.from_code(z).view(-1, 64, 8, 8)
        return self.decoder(h)
