"""
- object
mainファイル 学習実行するファイル

-detail
--input データのPATH
--out   出力先のPATH(学習モデル、予測結果)
--mode  0 -> spectrogram , 1 -> LLD
--hang  VAD にhang over処理を加えるか
"""

import os
from model import TimeActionPredict
from train import train_lstm

import torch.nn as nn
import torch.optim as optim

import argparse


def hang_over(y, flag=True):
    """
    u の末端 300 ms を １ にする
    """
    if flag:  
        for i in range(len(y)-1):
            if y[i] == 0 and y[i+1] == 1:
                y[i-3:i+1] = 1.
    return y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='/mnt/aoni04/katayama/DATA2020/')
    parser.add_argument('-m', '--mode', type=int,
                        help='mode is 0(spec) or 1(LLD)')
    parser.add_argument('-o', '--out', type=str, default='./SPEC')
    parser.add_argument('-e', '--epoch', type=int, default=100)
    parser.add_argument('-r', '--resume', type=str, default=True)
    parser.add_argument('--hang', type=str, default=False)

    args = parser.parse_args()
    import datetime
    now = datetime.datetime.now()
    print('{0:%Y%m%d%H%M}'.format(now))
    out = os.path.join(args.out, '{0:%Y%m%d%H%M}'.format(now))
    os.makedirs(out, exist_ok=True)

    dense_flag = False
    train_id = 89

    if args.mode == 0:
        from SPEC.utils import setup
        df_list = setup(PATH=args.input, dense_flag=dense_flag)
    else:
        from LLD.utils import setup
        df_list, lld_list = setup(PATH=args.input, dense_flag=dense_flag)
        lld_train = lld_list[13:train_id]
        lld_val = lld_list[train_id:]
        lld_dict = {'train': lld_train, 'val': lld_val}

    # 連結せずに 会話毎に list でもつ
    df_train = df_list[13:train_id]
    feature = []
    df_val = df_list[train_id:]
    feature_val = []
    df_dict = {'train': df_train, 'val': df_val}

    dataloaders_dict = {"train": feature, "val": feature_val}

    for phase in df_dict.keys():
        df = df_dict[phase]
        feature = dataloaders_dict[phase]

        for i in range(len(df)):
            if args.mode == 0:
                x = df[i].iloc[:, -512:-256].values
                x_b = df[i].iloc[:, -256:].values
            elif args.mode == 1:
                lld = lld_dict[phase]
                x = lld[i].iloc[:, :114].values
                x_b = lld[i].iloc[:, 114:].values
                x = x.reshape(-1, 10, 114)
                x_b = x_b.reshape(-1, 10, 114)
            u = hang_over(1.0 - df[i]['utter_A'].values, flag=args.hang)
            feature.append((x, u))
            u = hang_over(1.0 - df[i]['utter_B'].values, flag=args.hang)
            feature.append((x_b, u))        
        
    net = TimeActionPredict(
                input_size=x.shape[-1],
                hidden_size=64,
                mode=args.mode
    )                
    print('Model :', net.__class__.__name__)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for name, param in net.named_parameters():
        if 'fc' in name or 'lstm' in name:
            param.requires_grad = True
            print("勾配計算あり。学習する：", name)
        else:
            param.requires_grad = False
            print("勾配計算あり。学習しない：", name)
      
    train_lstm(net=net,
               dataloaders_dict=dataloaders_dict,
               criterion=criterion,
               optimizer=optimizer,
               num_epochs=args.epoch,
               output=out,
               resume=args.resume)

if __name__ == '__main__':
    main()
