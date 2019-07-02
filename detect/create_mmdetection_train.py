import os
import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import pickle
import glob

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
    #fn = os.path.join(args.img_dir, '{}.jpg'.format(image_id))
    fullpath = group['fullpath'].values[0]
    filename = group['filename'].values[0]

    assert image_id == os.path.basename(fullpath).split('.')[0]
    assert image_id == os.path.basename(filename).split('.')[0]

    width, height = get_image_size(fullpath)

    group['XMin'] = group['XMin'] * width
    group['XMax'] = group['XMax'] * width
    group['YMin'] = group['YMin'] * height
    group['YMax'] = group['YMax'] * height

    bboxes = [np.expand_dims(group[col].values, -1) for col in['XMin', 'YMin', 'XMax', 'YMax']]
    bboxes = np.concatenate(bboxes, axis=1)
    #print(bboxes)
    #print(bboxes.shape)
    return {
        'filename': group['filename'].values[0], #image_id+'.jpg',
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
    df = pd.read_csv(os.path.join(settings.DETECT_DATA_DIR, args.meta))
    #print(df.head())
    #files = sorted(os.listdir(args.img_dir))
    files = []
    for img_dir in args.img_dirs.split(','):
        files.extend(glob.glob(os.path.join(img_dir, '*.jpg')))
    print(files[:2])
    #fn_dict = { os.path.basename(x).split('.')[0]: os.path.basename(os.path.dirname(x)) + os.path.basename(x) for x in files}
    fn_dict = { os.path.basename(x).split('.')[0]: x for x in files}

    print('total train img files:', len(files))
    img_ids = [os.path.basename(x).split('.')[0] for x in files]
    print(img_ids[:2])

    annotation = df.loc[df['ImageID'].isin(set(img_ids))].copy()
    annotation['fullpath'] = annotation.ImageID.map(lambda x: fn_dict[x])
    if args.flat:
        annotation['filename'] = annotation['fullpath'].map(lambda x: os.path.basename(x))
    else:
        annotation['filename'] = annotation['fullpath'].map(lambda x: os.path.join(os.path.basename(os.path.dirname(x)), os.path.basename(x))) 

    groups = list(annotation.groupby('ImageID'))

    if n_samples > 0:
        groups = groups[:n_samples]
    #print('groups:', groups[:5])
    #print(group2mmdetection(groups[0]))

    with Pool(50) as p:
        samples = list(tqdm(iterable=p.imap_unordered(group2mmdetection, groups), total=len(groups)))

    print(samples[:2])

    out_file = os.path.join(settings.DETECT_DATA_DIR, args.output)
    with open(out_file, 'wb') as f:
        pickle.dump(samples, f)

def id2mmdetection(img_id):
    fn = os.path.join(args.test_img_dir, '{}.jpg'.format(img_id))
    width, height = get_image_size(fn)
    return {
        'filename': img_id+'.jpg',
        'width': width,
        'height': height,
    }

def create_test(args):
    df = pd.read_csv(os.path.join(settings.DETECT_DATA_DIR, 'sample_submission.csv'))
    with Pool(50) as p:
        img_ids = df.ImageId.values
        samples = list(tqdm(iterable=p.map(id2mmdetection, img_ids), total=len(img_ids)))

    out_file = os.path.join(settings.DETECT_DATA_DIR, args.output)
    with open(out_file, 'wb') as f:
        pickle.dump(samples, f)

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create mmdetection dataset')
    parser.add_argument('--img_dirs', type=str, default=None)
    parser.add_argument('--test_img_dir', type=str, default=None)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--flat', action='store_true')
    parser.add_argument('--meta', type=str, default='challenge-2019-train-detection-bbox.csv')
    args = parser.parse_args()

    if args.test:
        if args.test_img_dir is None:
            raise ValueError('test_img_dir')
        create_test(args)
    else:
        if args.img_dirs is None:
            raise ValueError('img_dirs')
        create_train(args)
