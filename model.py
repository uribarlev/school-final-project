import torch.nn as nn
class CNNEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(128, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )

    def forward(self, x):
        return self.conv(x)

class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = CNNEncoder()
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            bidirectional=True
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        feat = self.cnn(x)
        B, C, H, W = feat.size()
        feat = feat.mean(dim=2)
        feat = feat.permute(2, 0, 1)
        rnn_out, _ = self.rnn(feat)
        logits = self.fc(rnn_out)
        return logits, W