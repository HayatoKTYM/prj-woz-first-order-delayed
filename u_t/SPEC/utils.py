"""
- object
データセットを構築するファイル

- detail
spec_files .. スペクトログラム特徴量
feature_files .. その他(正解ラベルなど)

これらを pd.DataFrame 型で concat(axis=1) して返す
"""
import numpy as np
import pandas as pd
import glob
import os


def np_to_dataframe(np_list) -> pd.DataFrame:
    """
    np -> pd.DataFrame
    spectrogram file は 20fps 
    その他は 10fps で　あるため、フレームレートを 10 fpsに調整する
    """
    if type(np_list) == str: 
        np_list = np.load(np_list)
        np_list = np_list[:len(np_list)//2*2]  # 奇数個なら偶数個に
        np_list = np_list.reshape(-1, 256)  # 20fps > 10fps
        return pd.DataFrame(np_list)

      
def setup(PATH, dense_flag=False):
    spec_files = sorted(glob.glob(os.path.join(PATH, 'spec/*npy')))
    feature_files = sorted(glob.glob(os.path.join(PATH, 'feature/*csv')))
    print(f'file length is {len(spec_files)} and {len(feature_files)}')
    df_list = []

    for i in range(len(feature_files)):
        df = pd.read_csv(feature_files[i])
        img = np_to_dataframe(spec_files[2*i])
        imgB = np_to_dataframe(spec_files[2*i+1])
        
        df = df[:min([len(df), len(img)//2])]  # vad_file と長さ調整
        img = img[:min([len(df), len(img)])]
        imgB = imgB[:min([len(df), len(imgB)])]

        df = pd.concat([df, img, imgB], axis=1)
        df = df.fillna(0)
        df_list.append(df)

    return df_list
