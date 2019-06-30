import os
import pandas as pd
import numpy as np
import configs.settings as settings
from detect.utils import get_classes

def group2mmdetection(group: dict, stoi) -> dict:
    image_id, group = group
    height, width = 512, 512
    bboxes = [np.expand_dims(group[col].values, -1) for col in['XMin', 'YMin', 'XMax', 'YMax']]
    bboxes = np.concatenate(bboxes, axis=1)
    #print(bboxes)
    #print(bboxes.shape)
    return {
        'filename': image_id,
        'width': width,
        'height': height,
        'ann': {
            'bboxes': np.array(bboxes, dtype=np.float32),
            'labels': np.array([stoi[x] for x in group['LabelName'].values])
        }
    }

def create_train():
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
    classes, stoi = get_classes()

    n_samples = -1
    df = pd.read_csv(os.path.join(settings.DETECT_DATA_DIR, 'challenge-2019-train-detection-bbox.csv'))
    print(df.head())
    #files = sorted(os.listdir(settings.TRAIN_IMG_DIR))
    #print('total train img files:', len(files))
    annotation = df[:100000] # df.loc[df['ImageID'].isin(set(files))]

    groups = list(annotation.groupby('ImageID'))
    print('groups:', groups[:5])

    print(group2mmdetection(groups[0], stoi))

    #with Pool(args.n_jobs) as p:
    #    samples = list(tqdm(iterable=p.imap_unordered(group2mmdetection, groups), total=len(groups)))

    #with open(args.output, 'wb') as f:
    #    pickle.dump(samples, f)


if __name__ == '__main__':
    create_train()
