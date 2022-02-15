import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_class_names(df):
    labels = df['labels']
    return np.unique(' '.join(labels.unique()).split()).tolist()

def make_test_augmenter(conf):
    crop_size = round(conf.image_size*conf.crop_size)
    return  A.Compose([
        A.CenterCrop(height=crop_size, width=crop_size),
        A.Normalize(),
        ToTensorV2()
    ])

