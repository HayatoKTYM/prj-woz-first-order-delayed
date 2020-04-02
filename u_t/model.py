"""
- object
model 定義

-detail
spectrogram 入力か LLD 入力かは argument mode で 0 / 1 を指定
input_size は 256 / 114 になるのがdefault
hidden_size は ハイパーパラメータ
"""

import torch
import torch.nn as nn


class TimeActionPredict(nn.Module):
    """
    行動予測するネットワーク
    input_size ... 入力サイズ
    hidden_size ... 隠れ層のサイズ (FC, LSTM の次元)
    mode ... 0 -> spec , 1 -> lld
    """
    def __init__(self, input_size=256, hidden_size=64, mode=0):
        super(TimeActionPredict, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dr1 = nn.Dropout()
        self.relu1 = nn.ReLU()

        self.lstm = torch.nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.fc2 = nn.Linear(hidden_size, 2)      
        self.num_layers = 1
        self.hidden_size = hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode
        self.hidden = None

    def forward(self, x, middle=False):
        """
        x .... 入力(spectrogram)
        hidden ... LSTMの初期状態
        middle ... 戻り値に中間層出力を含めるか
        """
        if self.mode > 1 or self.mode < 0:
            print('error! please check argument mode is 0(spec mode) or 1(lld mode)') 
        elif self.mode == 0:
            x = self.dr1(self.relu1(self.fc1(x)))
            x = x.view(1, 1, -1)  # 2 -> 3
        else:
            fr, hs = x.size()
            x = self.dr1(self.relu1(self.fc1(x.view(-1, hs))))
            x = x.view(1, fr, -1)

        h, self.hidden = self.lstm(x, self.hidden)
        y = self.fc2(h[:, -1, :])

        if not middle:
            return y
        else:
            return y, h[:, -1, :]

    def reset_state(self):
        self.hidden = None

    def back_trancut(self):
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
