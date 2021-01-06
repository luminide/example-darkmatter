from imgaug import augmenters as iaa
import hp

def make_augmenters():
    sometimes = lambda aug: iaa.Sometimes(hp.aug_prob, aug)
    aug_list = [
        sometimes(iaa.Affine(rotate=(-20, 20), mode='symmetric')),
        iaa.CropToFixedSize(width=hp.crop_size, height=hp.crop_size),
        iaa.SomeOf(
            (0, hp.aug_count), [
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                sometimes(iaa.AddToHueAndSaturation(value=(-10, 10))),
                iaa.GaussianBlur((0, 3.0)),
            ],
            random_order=True
        )
    ]

    if hp.strong_aug:
        aug_list.append(
            iaa.SomeOf(
                (0, hp.aug_count), [
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
                    iaa.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                    iaa.Invert(0.05, per_channel=True),
                    iaa.Add((-10, 10), per_channel=0.5),
                    iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                    sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5)))
                ],
                random_order=True
            )
        )
    train_aug = iaa.Sequential(aug_list)

    test_aug = iaa.Sequential([
        iaa.CenterCropToFixedSize(width=hp.crop_size, height=hp.crop_size),
    ])

    return train_aug, test_aug
