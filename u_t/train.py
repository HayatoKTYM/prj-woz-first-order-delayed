"""
- object
学習の実行

-detail
spectrogram でも LLD でもコードは同一
train phase では 対話データをshuffleする
最終評価は、u_a と u_b を統合して評価する
"""

import torch
import torch.nn as nn

import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train(net, 
        dataloaders_dict, 
        criterion, optimizer,
        num_epochs=10,
        output='./',
        resume=False, 
        ):
    """
    学習ループ
    """
    #f = open('log.txt','w')
    os.makedirs(output,exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('using',device)
    net.to(device)

    Loss = {'train': [0]*num_epochs,
            'val': [0]*num_epochs}

    Acc = {'train': [0]*num_epochs,
            'val': [0]*num_epochs}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')
        # epochごとの訓練と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # モデルを訓練モードに
            else:
                net.eval()   # モデルを検証モードに

            epoch_loss = 0.0  # epochの損失和
            epoch_corrects = 0  # epochの正解数
            
            for inputs, labels in dataloaders_dict[phase]:    
                inputs = inputs.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)
                
                out = net(inputs)  # 順伝播
                loss = criterion(out, labels.view(-1))  # ロスの計算
                _, preds = torch.max(out, 1)  # ラベルを予測
                
                if phase == 'train':  # 訓練時はバックプロパゲーション
                    optimizer.zero_grad()  # 勾配の初期化
                    loss.backward()  # 勾配の計算
                    optimizer.step()  # パラメータの更新

                epoch_loss += loss.item()
                epoch_corrects += torch.sum(preds == labels.data)

                
            # epochごとのlossと正解率を表示
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            Loss[phase][epoch] = epoch_loss 
            Acc[phase][epoch] = float(epoch_acc.cpu().numpy())
            
        if resume:
            torch.save(net.state_dict(), os.path.join(output,'epoch_{}_acc{:.2f}_loss{:.2f}_ut_train.pth'.format(epoch+1,epoch_acc,epoch_loss)))
        
            y_true, y_prob = np.array([]), np.array([])
            for inputs, labels in dataloaders_dict['test']:
                inputs = inputs.to(device,dtype=torch.float32)
                out = net(inputs)
                _, preds = torch.max(out, 1)  # ラベルを予測
                
                y_true = np.append(y_true, labels.data.numpy())
                y_prob = np.append(y_prob, nn.functional.softmax(out,dim=-1).cpu().data.numpy()[:,1])

            # u_a_(t) と u_b_(t) から　u(t) を算出
            y_a = y_true[:len(y_true)//2]
            y_b = y_true[len(y_true)//2:]
            y_true = [min([y_a[i],y_b[i]]) for i in range(len(y_a))]
            y_true = np.clip(y_true,0,1)
            
            y_a = y_prob[:len(y_prob)//2]
            y_b = y_prob[len(y_prob)//2:]
            y_prob = [min([y_a[i],y_b[i]]) for i in range(len(y_a))]
            
            #保存
            plt.figure(figsize=(20,4))
            plt.rcParams["font.size"] = 18
            plt.plot(y_true[:300],label = 'true label',color='r',linewidth=3.0)
            plt.plot(y_prob[:300],label = 'predict',color='m')
            plt.fill_between(list(range(300)),y_prob[:300],color='m',alpha=0.35)
            plt.legend()
            plt.savefig(os.path.join(output,'result_{}_acc{}_loss{}_ut_train.png'.format(epoch+1,epoch_acc,epoch_loss)))

        #print(confusion_matrix(y_true, y_pred))
        #print(classification_report(y_true, y_pred))
        print('-------------')
            
    if resume: # 学習過程(Loss) を保存するか    
        plt.figure(figsize=(15,4))
        plt.plot(Loss['val'],label='val')
        plt.plot(Loss['train'],label='train')
        plt.legend()
        plt.savefig(os.path.join(output,'history.png'))

    print('training finish and save train history...')
    y_true, y_pred = np.array([]), np.array([])
    y_prob = np.array([])

    for inputs, labels in dataloaders_dict['test']:
        inputs = inputs.to(device,dtype=torch.float32)
        out = net(inputs)
        _, preds = torch.max(out, 1)  # ラベルを予測

        y_true = np.append(y_true, labels.data.numpy())
        y_pred = np.append(y_pred, preds.cpu().data.numpy())
        y_prob = np.append(y_prob, nn.functional.softmax(out).cpu().data.numpy()[:,1])

    print(accuracy_score(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))
    
    # u_a_(t) と u_b_(t) から　u(t) を算出
    y_a = y_true[:len(y_true)//2]
    y_b = y_true[len(y_true)//2:]
    y_true = [min([y_a[i],y_b[i]]) for i in range(len(y_a))]
    y_true = np.clip(y_true,0,1)
    
    y_a = y_prob[:len(y_prob)//2]
    y_b = y_prob[len(y_prob)//2:]
    y_prob = [min([y_a[i],y_b[i]]) for i in range(len(y_a))]
    
    if resume: # 出力の可視化例を保存するかどうか
        plt.figure(figsize=(15,4))
        plt.plot(y_true[:300],label = 'true label')
        plt.plot(y_prob[:300],label = 'predict')
        plt.legend()
        plt.savefig(os.path.join(output,'result.png'))

def train_lstm(net, 
        dataloaders_dict, 
        criterion, optimizer,
        num_epochs=10,
        output='./',
        resume=True,         
    ):
    """
    学習ループ
    """
    os.makedirs(output,exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('using',device)
    net.to(device)

    Loss = {'train': [0]*num_epochs,
            'val': [0]*num_epochs}

    Acc = {'train': [0]*num_epochs,
            'val': [0]*num_epochs}
    fo = open(os.path.join(output,'log.txt'),'a')
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')
        # epochごとの訓練と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0
            hidden = None
            loss = 0
            train_cnt = 0
            total = 0.0
            feature = np.array(dataloaders_dict[phase])
            #会話データのシャッフル(loss 計算する会話データの順番を変えるため)
            if phase == 'train':
                N = np.random.permutation(len(feature))
                print(N)
            else:
                N = np.arange(len(feature))

            for f in feature[N]:
                hidden = None
                net.reset_state()
                total += len(f[0])
                for i in range(len(f[0])):
                    inputs = torch.tensor(f[0][i]).to(device,dtype=torch.float32)
                    labels = torch.tensor(f[1][i]).to(device,dtype=torch.long)
                    #print(labels)
                    out = net(inputs)
                    train_cnt += 1
                    l = criterion(out, labels.view(-1))
                    loss += l
                    _, preds = torch.max(out, 1)
                
                    if phase == 'train' and train_cnt % 32 == 0 : # 訓練時はバックプロパゲーション
                        optimizer.zero_grad() # 勾配の初期化
                        loss.backward() # 勾配の計算
                        optimizer.step()# パラメータの更新
                        loss = 0 #累積誤差reset
                        net.back_trancut()

                    epoch_loss += l.item()  # lossの合計を更新
                    epoch_corrects += torch.sum(preds == labels.data) # 正解数の合計を更新
                
            # epochごとのlossと正解率を表示
            epoch_loss = epoch_loss / total
            epoch_acc = epoch_corrects.double() / total
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            Loss[phase][epoch] = epoch_loss 
            Acc[phase][epoch] = float(epoch_acc.cpu().numpy())
            
        if resume:
            torch.save(net.state_dict(), os.path.join(output,'epoch_{}_acc{:.3f}_loss{:.3f}_ut_train.pth'.format(epoch+1,epoch_acc,epoch_loss)))
            y_true, y_prob = np.array([]), np.array([])
            y_pred = np.array([])
            feature = np.array(dataloaders_dict['val'])
            #feature = feature[::2] + feature[1::2] # u_a と u_b の or をとって評価するため順序を入れ替え 
            for f in feature[::2]:
                net.reset_state()
                for i in range(len(f[0])):
                    inputs = torch.tensor(f[0][i]).to(device,dtype=torch.float32)
                    labels = torch.tensor(f[1][i]).to(device,dtype=torch.long)
                    out = net(inputs)
                    #print(labels)

                    _, preds = torch.max(out, 1)  # ラベルを予測
                    y_pred = np.append(y_pred, preds.cpu().data.numpy())
                    y_true = np.append(y_true, labels.cpu().data.numpy())
                    y_prob = np.append(y_prob, nn.functional.softmax(out,dim=-1).cpu().data.numpy()[0][1])
            for f in feature[1::2]:
                net.reset_state()
                for i in range(len(f[0])):
                    inputs = torch.tensor(f[0][i]).to(device,dtype=torch.float32)
                    labels = torch.tensor(f[1][i]).to(device,dtype=torch.long)
                    out = net(inputs)
                    
                    _, preds = torch.max(out, 1)  # ラベルを予測
                    y_pred = np.append(y_pred, preds.cpu().data.numpy())
                    y_true = np.append(y_true, labels.cpu().data.numpy())
                    y_prob = np.append(y_prob, nn.functional.softmax(out,dim=-1).cpu().data.numpy()[0][1])
            
            print(classification_report(y_true, y_pred))
            print(classification_report(y_true, y_pred),file=fo)
            # u_a_(t) と u_b_(t) から　u(t) を算出
            y_a = y_true[:len(y_true)//2]
            y_b = y_true[len(y_true)//2:]
            y_true = [min([y_a[i],y_b[i]]) for i in range(len(y_a))]
            y_true = np.clip(y_true,0,1)
            
            y_a = y_prob[:len(y_prob)//2]
            y_b = y_prob[len(y_prob)//2:]
            y_prob = [min([y_a[i],y_b[i]]) for i in range(len(y_a))]
            
            plt.figure(figsize=(20,4))
            plt.rcParams["font.size"] = 18
            plt.plot(y_true[:300],label = 'true label',color='r',linewidth=3.0)
            plt.plot(y_prob[:300],label = 'predict', color='m')
            plt.fill_between(list(range(300)),y_prob[:300],color='m',alpha=0.35)
            plt.legend()
            plt.savefig(os.path.join(output,'result_{}_acc{:.3f}_loss{:.3f}_ut_train.png'.format(epoch+1,epoch_acc,epoch_loss)))

        print('-------------')
    fo.close()

    if resume: # 学習過程(Loss) を保存するか    
        plt.figure(figsize=(15,4))
        plt.rcParams["font.size"] = 15
        plt.plot(Loss['val'],label='val')
        plt.plot(Loss['train'],label='train')
        plt.legend()
        plt.savefig(os.path.join(output,'history.png'))
