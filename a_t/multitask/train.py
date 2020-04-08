import sys
sys.path.append('../')
from eval import quantitative_evaluation
import torch
import torch.nn as nn
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import matplotlib
import matplotlib.pyplot as plt


def train(net, 
        dataloaders_dict, 
        criterion, optimizer,
        num_epochs=10,
        output='./',
        resume=False, 
        ):
    """
    学習ループ
    0. for n in epoch:
    1.会話データをシャッフル
    2.for Train 会話データ:
        reset_state
        3.for 1会話:
            loss の計算など
    4.for val 会話データ:
        評価
    """
    os.makedirs(output,exist_ok=True)
    Loss = {'train':[],'val':[]}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('using',device)
    net.to(device)
    u_t_criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # モデルを訓練モードに
            else:
                net.eval()   # モデルを検証モードに

            epoch_loss = 0.0  # epochの損失和
            epoch_loss2 = 0.0  # epochの損失和
            epoch_corrects = 0.0
            total = 0
            loss = 0
            train_cnt = 0
            frame_cnt = 0
            threshold = 0.8
            back_cnt = 0
            calc_flag = True 

            feature = np.array(dataloaders_dict[phase])
            #会話データのシャッフル(loss 計算する会話データの順番を変えるため)
            if phase == 'train':
                N = np.random.permutation(len(feature))
                print(N)
            else:
                N = np.arange(len(feature))

            for f in feature[N]:
                hidden = None
                out = 0
                a = 0
                net.reset_state_ut()
                total += len(f[0])
                for i in range(len(f[0])):
                    inputs = torch.tensor(f[0][i]).to(device, dtype=torch.float32)
                    labels = torch.tensor(f[2][i]).to(device, dtype=torch.float32)
                    u_a_label = torch.tensor(f[3][i]).to(device, dtype=torch.long) 
                    u_b_label = torch.tensor(f[4][i]).to(device, dtype=torch.long) 
                    frame_cnt += 1
                    out, a, u, u_a, u_b = net(inputs, 0, out, a)
                    l = u_t_criterion(u_a, u_a_label.view(-1))
                    _, preds = torch.max(u_a, 1)
                    epoch_corrects += torch.sum(preds == u_a_label.data)
                    loss += l
                    epoch_loss2 += l.item()
                    l = u_t_criterion(u_b, u_b_label.view(-1))
                    _, preds = torch.max(u_b, 1)
                    epoch_corrects += torch.sum(preds == u_b_label.data)
                    
                    loss += l
                    epoch_loss2 += l.item()
                    
                    if labels >= threshold:
                        l = 15 * criterion(out, labels)
                        loss += (l * frame_cnt)
                        epoch_loss += l.item()
                        train_cnt += 1
                        back_cnt += 1

                    if out < threshold: # 計算フラグ初s期化
                        calc_flag = True

                    elif out >= threshold and calc_flag: # 1回計算したらそのあとのyt > threshold は計算しない
                        l = 15  * criterion(out, out.data * 0.8) # ロスの計算
                        loss += (l * frame_cnt) 
                        epoch_loss += l.item()
                        train_cnt += 1
                        back_cnt += 1
                        calc_flag = False

                    if phase == 'train' and back_cnt == 1:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        loss = 0
                        back_cnt = 0
                        frame_cnt = 0
                        net.back_trancut_ut()
                        out = out.detach()
                        a = a.detach()            
            print(epoch_corrects, total)
            epoch_loss = epoch_loss / train_cnt
            epoch_loss2 = epoch_loss2 / total / 2
            epoch_corrects = epoch_corrects.double() / total / 2
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            print('{} Loss2: {:.4f}'.format(phase, epoch_loss2))
            print('{} u(t) Acc: {:.3f}'.format(phase, epoch_corrects))
            Loss[phase].append(epoch_loss)

        if resume:
            torch.save(net.state_dict(), os.path.join(output,'epoch_{}_loss_{:.3f}_ut_train.pth'.format(epoch+1,epoch_loss)))
            y_true, y_pred = np.array([]), np.array([])
            a_values_list = np.array([])
            u_values_list = np.array([])
            u_true = np.array([])
            feature = np.array(dataloaders_dict['val'])
            for f in feature:
                out = 0
                a = 0
                y_true = np.append(y_true, f[2])
                net.reset_state_ut()
                for i in range(len(f[0])):
                    inputs = torch.tensor(f[0][i]).to(device, dtype=torch.float32)
                    u_val = torch.tensor(f[1][i])
                    labels = torch.tensor(f[2][i], dtype=torch.long)
                    
                    out, a, u, u_a, u_b = net(inputs, u_val, out, a)
                    y_pred = np.append(y_pred, out.cpu().data.numpy())
                    a_values_list = np.append(a_values_list, a.cpu().data.numpy())
                    u_values_list = np.append(u_values_list, u.cpu().data.numpy())
                    u_true = np.append(u_true, u_val.data.numpy())

            plt.figure(figsize=(20, 4))
            plt.rcParams["font.size"] = 18
            plt.plot([threshold]*300,color='black',linestyle='dashed')
            plt.plot(y_pred[:300],label = 'predict',linewidth=3.0)
            
            plt.plot(u_values_list[:300],label='u_t',color='#B8DBB9',linewidth=2.0)
            plt.fill_between(range(300),u_values_list[:300],color='g',alpha=0.3)
            plt.plot(u_true[:300],label='u_true',color='g',linewidth=3.0)
            plt.plot(a_values_list[:300],label='a_t',color='r',linewidth=3.0)
            plt.fill_between(range(300),a_values_list[:300],color='r',alpha=0.3)
            plt.plot(y_true[:300],label = 'true label',linewidth=4.0,color='m')
            plt.legend()
            plt.savefig(os.path.join(output,'result_{}_loss_{:.3f}.png'.format(epoch+1,epoch_loss)))

            precision, recall, f1, Distance = quantitative_evaluation(y_true, y_pred, u_true, resume=True, output=output)
        print('-------------')

    if resume: # 学習過程(Loss) を保存するか
        plt.figure(figsize=(15,4))
        plt.plot(Loss['val'],label='val')
        plt.plot(Loss['train'],label='train')
        plt.legend()
        plt.savefig(os.path.join(output,'history.png'))
