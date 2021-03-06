"""
-object
時定数a(t) をよそくする モデルの定義

-abstract
時定数 a(t) と 非発話らしさ u(t) から 行動生成らしさを計算する
a(t) は 入力から FC 層で 1dim に圧縮して決める
u(t) は 離散値(真値) を　使う場合と 連続値(予測値) を使う場合がある為、
1. FirstDelayActionPredict_ut_model .. 連続値モデル
2. FirstDelayActionPredict_VAD .. 離散値モデル
を定義

-detail
FirstDelayActionPredict_ut_model
    num_layers .. 1 (LSTM のスタック数)
    input_size .. 入力サイズ
    hidden_size .. 隠れ層のサイズ
    PATH .. 事前学習済み u(t) よそくモデルの重み
    lstm_model .. u(t) モデルの選択 , True だと LSTM model
    mode .. User 毎の特徴量を concat / add

FirstDelayActionPredict_VAD
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../')
from u_t.model import *


class FirstDelayActionPredict_ut_model(nn.Module):
    """
    行動予測するネットワーク
    時定数 a(t) を　学習する
    u(t) .. 非発話らしさ (step応答)
    a(t),u(t) から 1次遅れ系のstep応答を計算し， 行動生成らしさ y(t) を計算
    u_t は予測値を与える(0 ~ 1 の連続値)
    """
    def __init__(self, 
                 num_layers=1, 
                 input_size=128, 
                 hidden_size=64, 
                 PATH='../../u_t_dense/dense_model/epoch_29_ut_train.pth',
                 lstm_model=False,
                 mode='concat'
                 ):
        super(FirstDelayActionPredict_ut_model, self).__init__()

        # u(t) を frame by frame で予測
        if not lstm_model:
            self.u_t_model = U_t_train()
            self.u_t_model.load_state_dict(torch.load(PATH,map_location='cpu'))
        # u(t) を LSTM を用いた時系列モデルで予測
        else:
            self.u_t_model = TimeActionPredict()
            self.u_t_model.load_state_dict(torch.load(PATH,map_location='cpu'))

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dr1 = nn.Dropout()
        self.relu1 = nn.ReLU()
        self.lstm = torch.nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.fc3 = nn.Linear(hidden_size, 1)

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.count = 0
        self.hidden = None
        self.h_a = None
        self.h_b = None
        self.mode = mode
        self.b_cnt = 0

        self.u_t_model.to(self.device)
        self.u_t_model.eval()


    def forward(self, x, u=0, y_pre=0, a_pre=0):
        """
         u は不使用
        """
        x = x.unsqueeze(0)
        ########### u(t) BLOCK ###############
        u_a, h_u_a = self.u_t_model(
                                x=x[:, -512:-256],
                                middle=True)

        u_b, h_u_b = self.u_t_model(
                                x=x[:, -256:],
                                middle=True)

        u_a = F.softmax(u_a)
        u_b = F.softmax(u_b)
        u = torch.min(torch.cat((u_a[:, 1],u_b[:, 1]), dim=-1))
        ########################################
        #  h_u_a と h_u_b をconat
        if self.mode == 'concat':
            h = torch.cat((h_u_a, h_u_b, x[:, :-512]), dim=-1)
        elif self.mode == 'add':
            h = torch.add(h_u_a, h_u_b)

        ########### a(t) BLOCK ###############
        x = self.dr1(self.relu1(self.fc1(x)))
        x = x.view(1, 1, -1)

        h, self.hidden = self.lstm(x, self.hidden)
        a = F.sigmoid(self.fc3(h[:,-1,:]))
        ########################################

        ########### y(t) BLOCK ###############
        a = u * a_pre + (1-u) * a
        # if self.count % 50000 == 0:
        #     print("u:{}, a:{}".format(u, a))
        self.count += 1
        y = a * u + (1-a) * y_pre
        if u < 0.5:
            y -= y.data
        ######################################
        return y, a, u

    def reset_state(self):
        self.hidden = None
    
    def back_trancut(self):
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())

    def reset_state_ut(self):
        self.u_t_model.reset_state()

    def back_trancut_ut(self):
        self.u_t_model.back_trancut()
        if self.count % 50000 == 0:
            print('detached!')


class FirstDelayActionPredict_VAD(nn.Module):
    """
    行動予測するネットワーク
    時定数 a(t) を　学習する
    u(t) .. 非発話らしさ (step応答)
    a(t),u(t) から 1次遅れ系のstep応答を計算し， 行動生成らしさ y(t) を計算
    u(t) は 0 / 1 の離散値を入力する version
    """
    def __init__(self, 
                 num_layers=1, 
                 input_size=128, 
                 hidden_size=64, 
                 mode='concat'):
        super(FirstDelayActionPredict_VAD, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dr1 = nn.Dropout()
        self.relu1 = nn.ReLU()

        self.lstm = torch.nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.fc3 = nn.Linear(hidden_size, 1)

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.count = 0
        self.hidden = None
        self.mode = mode

    def forward(self, x, u=0, y_pre=0, a_pre=0):
        x = x.unsqueeze(0)
        u = u.unsqueeze(0)
        if 'add' in self.mode:
            x_ = torch.add(x[:, :256], x[:, 256:-1])
            x = torch.cat((x_, x[:, -1].reshape(-1, 1)), dim=1)
        x = self.dr1(self.relu1(self.fc1(x)))
        x = x.view(1, 1, -1)

        h, self.hidden = self.lstm(x, self.hidden)
        a = F.sigmoid(self.fc3(h[:, -1, :]))
        a = u * a_pre + (1-u) * a
        
        y = a * u + (1-a) * y_pre
        if u <= 0.0:
            y -= y.data

        return y, a, u

    def reset_state(self):
        self.hidden = None
    
    def back_trancut(self):
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())

    def reset_state_ut(self):
        """
        冗長性
        """
        pass

    def back_trancut_ut(self):
        """
        冗長性
        """
        pass
