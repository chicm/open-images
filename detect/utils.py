import os
import pandas as pd
import configs.settings as settings


def get_classes():
    df = pd.read_csv(os.path.join(settings.DETECT_DATA_DIR, 'challenge-2019-classes-description-500.csv'), header=None, names=['classes', 'description'])
    c = df.classes.values
    print(df.head())
    stoi = { c[i]: i for i in range(len(c)) }
    return c, stoi
