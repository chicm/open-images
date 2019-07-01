import os
import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import pickle

import configs.settings as settings
from detect.utils import get_classes, get_image_size

_, stoi = get_classes()
args = None

def group2mmdetection(group: dict) -> dict:
    """Custom dataset for detection.

    Annotation format:
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann': {
                'bboxes': <np.ndarray> (n, 4),
                'labels': <np.ndarray> (n, ),
                'bboxes_ignore': <np.ndarray> (k, 4),
                'labels_ignore': <np.ndarray> (k, 4) (optional field)
            }
        },
        ...
    ]

    The `ann` field is optional for testing.
    """

    image_id, group = group
    fn = os.path.join(args.img_dir, '{}.jpg'.format(image_id))
    width, height = get_image_size(fn)

    group['XMin'] = group['XMin'] * width
    group['XMax'] = group['XMax'] * width
    group['YMin'] = group['YMin'] * height
    group['YMax'] = group['YMax'] * height

    bboxes = [np.expand_dims(group[col].values, -1) for col in['XMin', 'YMin', 'XMax', 'YMax']]
    bboxes = np.concatenate(bboxes, axis=1)
    #print(bboxes)
    #print(bboxes.shape)
    return {
        'filename': image_id+'.jpg',
        'width': width,
        'height': height,
        'ann': {
            'bboxes': np.array(bboxes, dtype=np.float32),
            'labels': np.array([stoi[x] for x in group['LabelName'].values]) + 1
        }
    }

def create_train(args):
    classes, stoi = get_classes()

    n_samples = -1
    df = pd.read_csv(os.path.join(settings.DETECT_DATA_DIR, 'challenge-2019-train-detection-bbox.csv'))
    #print(df.head())
    files = sorted(os.listdir(args.img_dir))
    print('total train img files:', len(files))

    img_ids = [os.path.basename(x).split('.')[0] for x in files]

    annotation = df.loc[df['ImageID'].isin(set(img_ids))]

    groups = list(annotation.groupby('ImageID'))

    if n_samples > 0:
        groups = groups[:n_samples]
    #print('groups:', groups[:5])
    #print(group2mmdetection(groups[0]))

    with Pool(50) as p:
        samples = list(tqdm(iterable=p.imap_unordered(group2mmdetection, groups), total=len(groups)))

    out_file = os.path.join(settings.DETECT_DATA_DIR, args.output)
    with open(out_file, 'wb') as f:
        pickle.dump(samples, f)

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create mmdetection dataset')
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    create_train(args)
