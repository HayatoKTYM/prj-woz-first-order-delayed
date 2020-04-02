"""
- object
データセットを構築するファイル

- detail
lld_files .. LLD特徴量
feature_files .. その他(正解ラベルなど)

これらを pd.DataFrame 型で concat(axis=1) して返す
"""
import pandas as pd
import glob
import os


def setup(PATH, dense_flag=False):
    """
    np -> pd.DataFrame
    LLD file は 100fps 
    その他は 10fps で　あるため、フレームレートを 10 fpsに調整する
    """
    lld_files = sorted(glob.glob(os.path.join(PATH, 'lld_all/*csv')))
    feature_files = sorted(glob.glob(os.path.join(PATH, 'feature/*csv')))
    print(f'file length is {len(lld_files)} and {len(feature_files)}')
    df_list = []
    lld_list = []
    for i in range(len(feature_files)):
        df = pd.read_csv(feature_files[i])
        try:
            lld = pd.read_csv(lld_files[i])
        except:
            print(lld_files[i])
            continue
        length = min([len(df), len(lld)//10])
        df = df[:length]
        df_list.append(df)
        lld = lld[:length*10]
        lld_list.append(lld)

    return df_list, lld_list
