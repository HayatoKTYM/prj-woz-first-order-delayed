import pandas as pd
import numpy as np
import sys
from torch.utils import data
import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from train import train
from model import FirstDelayActionPredict_ut_model
sys.path.append('../')
from utils import setup

def hang_over(y):
    """
    u の末端 200 ms を １ にする
    """
    for i in range(len(y)-1):
        if y[i] == 0 and y[i+1] == 1:
            y[i-2:i+1] = 1.
    return y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='/mnt/aoni04/katayama/DATA2020/')
    parser.add_argument('-o', '--out', type=str, default='./DISCRETE')
    parser.add_argument('-w', '--u_PATH', type=str,
    default='../u_t/SPEC/SPEC/202004021729/epoch_21_acc0.909_loss0.218_ut_train.pth')
    parser.add_argument('-e', '--epoch', type=int, default=100)
    parser.add_argument('-r', '--resume', type=str, default=True)
    parser.add_argument('--hang', type=str, default=False)

    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import datetime
    now = datetime.datetime.now()
    print('{0:%Y%m%d%H%M}'.format(now))
    out = os.path.join(args.out, '{0:%Y%m%d%H%M}'.format(now))
    os.makedirs(out, exist_ok=True)

    df_list = setup(PATH=args.input, dense_flag=False)
    train_id = 100
    # 連結せずに 会話毎に list でもつ
    df_train = df_list[13:train_id]
    feature = []
    df_val = df_list[train_id:]
    feature_val = []
    
    df_dict = {'train':df_train, 'val':df_val}
    dataloaders_dict = {"train": feature, "val": feature_val}

    for phase in df_dict.keys():
        df = df_dict[phase]
        feature = dataloaders_dict[phase]

        for i in range(len(df)):
            x = df[i].iloc[:,-512:].values
            u = np.clip(1.0 - (df[i]['utter_A'] + df[i]['utter_B'] ),0,1)
            u_a = 1.0 - df[i]['utter_A'].values
            u_b = 1.0 - df[i]['utter_B'].values
            #print(df[i]['utter_A'].values)
            target = df[i]['target'].map(lambda x:0 if x == 'A' else 1).values
            target = target.reshape(len(target),1)
            y = df[i]['action'].map(lambda x:0.8 if x == 'Passive' else 0.8 if x == 'Active' else 0).values
            y[u==0] = 0.
            y[0] = 0.0 # start は 予測不能
            #x = np.append(target,x,axis=1)
            feature.append((x,u,y,u_a,u_b))

    net = FirstDelayActionPredict_ut_model(
            input_size=512,#+64+1,
            hidden_size=128,
            PATH=args.u_PATH,
            lstm_model=True)
    print('Model :', net.__class__.__name__)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    for name, param in net.named_parameters():
        param.requires_grad = True
        print("勾配計算あり。学習する：", name)
        
    print(len(x)==len(y))
    assert len(x) == len(y); print('problem occurred!! please check your dataset length..')
    print('train data is ',np.shape(dataloaders_dict['train']))
    print('test data is ',np.shape(dataloaders_dict['val']))

    train(
        net=net, 
        dataloaders_dict=dataloaders_dict, 
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.epoch, 
        output=out,
        resume=args.resume)


if __name__ == '__main__':
    main()
