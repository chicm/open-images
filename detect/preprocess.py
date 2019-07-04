import os
import pandas as pd
import numpy as np

import configs.settings as settings

DATA_DIR = settings.DETECT_DATA_DIR


def gen_top_classes():
    df_train = pd.read_csv(os.path.join(DATA_DIR, 'challenge-2019-train-detection-bbox.csv'))
    #print(df_train.head())

    df_counts = pd.DataFrame(df_train.LabelName.value_counts())
    df_counts.index.name = 'class'
    df_counts.columns =['count']
    print(df_counts.head())
    df_counts.to_csv(os.path.join(DATA_DIR,'top_classes.csv'), index=True)

def check_data(n_split = 100):
    df_train = pd.read_csv(os.path.join(DATA_DIR, 'challenge-2019-train-detection-bbox.csv'))
    print(df_train.head())

    num_objs = df_train.LabelName.value_counts().values
    top_classes = df_train.LabelName.value_counts().index
    print(num_objs[:n_split].sum() / num_objs.sum())

    print(len(df_train.ImageID.unique()))
    df_top = df_train.loc[df_train.LabelName.isin(set(top_classes[:n_split]))]
    print(len(df_top.ImageID.unique()))

    df_bottom = df_train.loc[df_train.LabelName.isin(set(top_classes[n_split:]))]
    print(len(df_bottom.ImageID.unique()))

if __name__ == '__main__':
    check_data(100)
    #gen_top_classes()

