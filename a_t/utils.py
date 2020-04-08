"""
-object
データセットを構築するファイル

-detail
setup  ファイルの読み込み
       feature/*csv         wizard の操作ログや VAD ラベルなど
       img_middle64/*npy    画像特徴量の中間出力 [64dim]
       spec/*npy   音響特徴量の中間出力 [256dim × 2 users]
"""
import numpy as np
import pandas as pd
import glob
import os


def np_to_dataframe(np_list) -> pd.DataFrame:
    """
    np -> pd.DataFrame
    """
    if type(np_list) == str:
        np_list = np.load(np_list)
        np_list = np_list[:len(np_list)//2*2] #奇数個なら偶数個に
        np_list = np_list.reshape(-1,256) #20fps > 10fps
        return pd.DataFrame(np_list)
    else: #np.load 済みなら
        return pd.DataFrame(np_list)


def add_active(df, file_name):
    file_name = file_name.replace('/feature', '/TEXT', 1).replace('.feature.csv', '.vad.txt', 1)
    try:
        with open(file_name) as f:
            lines = f.readlines()
            #lines = [line.rstrip().split('\t') for line in lines]
            for line in lines:
                act, _, start, end, _ = line.rstrip().split('\t')
                start = int(start) // 100
                if act != 'default':
                    break
                #print(df['action'].value_counts())
                df['action'].values[start] = 'Active'
                #print(df['action'].value_counts())
    except FileNotFoundError:
        print(file_name)

    return df


def setup(PATH='/mnt/aoni04/katayama/DATA2020', dense_flag=False, elan_flag=False):
    gaze_files = sorted(glob.glob(os.path.join(PATH, 'img_middle64/*npy')))
    img_middle_feature_files = sorted(glob.glob(os.path.join(PATH, 'spec/*npy')))
    feature_files = sorted(glob.glob(os.path.join(PATH, 'feature/*csv')))
    
    print(f'file length is {len(img_middle_feature_files)} and {len(feature_files)} and {len(gaze_files)}')
    df_list = []

    for i in range(len(feature_files)):
        df = pd.read_csv(feature_files[i])
        if elan_flag:
            df = add_active(df, feature_files[i])

        gaze = pd.DataFrame(np.load(gaze_files[i]))
        img = np_to_dataframe(img_middle_feature_files[2*i])
        imgB = np_to_dataframe(img_middle_feature_files[2*i+1])

        df = df[:min([len(df), len(img), len(gaze)])]  # vad_file と長さ調整
        img = img[:min([len(df), len(img), len(gaze)])]
        imgB = imgB[:min([len(df), len(imgB), len(gaze)])]

        df = pd.concat([df, gaze, img, imgB], axis=1)
        df = df.fillna(0)
        df_list.append(df)

    return df_list
