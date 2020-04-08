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

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../../')
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

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('using',device)
        self.u_t_model.to(device)
        self.u_t_model.eval()


        self.fc3 = nn.Linear(hidden_size, 1)

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.count = 0
        self.h_a = None
        self.h_b = None
        self.b_cnt = 0

    def forward(self, x, u=0, y_pre=0, a_pre=0):
        x = x.unsqueeze(0)
        
        u_a, h_u_a = self.u_t_model(
                                x=x[:, -512:-256],
                                middle=True)

        u_b, h_u_b = self.u_t_model(
                                x=x[:, -256:],
                                middle=True)
        u_a = F.softmax(u_a)
        u_b = F.softmax(u_b)
        u = torch.min(torch.cat((u_a[:,1],u_b[:,1]),dim=-1))
        
        #  h_u_a と h_u_b をconat
        h = torch.cat((h_u_a, h_u_b, x[:,:-512]), dim=-1)
        a = F.sigmoid(self.fc3(h))
        a = u * a_pre + (1-u) * a
        #if self.count % 50000 == 0:
        #    print("u:{}, a:{}".format(u,a))
        self.count += 1
        y1 = a * u + (1-a) * y_pre
        if u < 0.5:
            y1 -= y1
        return y1, a, u, u_a, u_b

    def reset_state(self):
        pass
    
    def back_trancut(self):
        pass

    def reset_state_ut(self):
        self.u_t_model.reset_state()

    def back_trancut_ut(self):
        self.u_t_model.back_trancut()
        if self.count % 50000 == 0:
            print('detached!')