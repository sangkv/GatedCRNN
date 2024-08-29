import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.gate = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    
    def forward(self, x):
        gate_output = torch.sigmoid(self.gate(x))
        output = x * gate_output
        return output


class Encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(2, 4), stride=1, padding=(1, 2))
        self.gate2 = GatedConv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.gate3 = GatedConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(2, 4), stride=1, padding=(0, 1))
        self.gate4 = GatedConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = self.gate2(x)
        x = torch.tanh(self.conv3(x))
        x = self.gate3(x)
        x = torch.tanh(self.conv4(x))
        x = self.gate4(x)
        x = torch.tanh(self.conv5(x))
        return x


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, hidden_size)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, bidirectional=True, batch_first=True)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = torch.tanh(self.linear(x))
        x, _ = self.lstm2(x)
        return x


class GatedCRNN(nn.Module):
    def __init__(self, num_classes, input_channel=1, hidden_size=128):
        super().__init__()
        self.encoder = Encoder(input_channel)
        self.decoder = Decoder(input_size=128, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input):
        visual_feature = self.encoder(input)
        b, c, h, w = visual_feature.size()
        visual_feature = F.max_pool2d(visual_feature, kernel_size=(h, 1), stride=(h, 1))
        visual_feature = visual_feature.permute(0, 3, 1, 2).contiguous().view(b, w, -1)

        contextual_feature = self.decoder(visual_feature)

        prediction = self.fc(contextual_feature)
        return prediction
