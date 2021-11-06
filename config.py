default_config = dict(
    # this can be any network from the timm library
    arch = 'resnet18',

    pretrained = True,

    crop_width = 128,
    crop_height = 128,

    # optimizer settings
    lr = 0.01,
    momentum = 0,
    nesterov = False,
    batch_size = 16,

    # scheduler settings
    gamma = 0.96,

    # images will be resized to margin*4% larger than crop size
    margin = 3,
    aug_prob = 0.75,
    strong_aug = True,
)

class Config():
    def __init__(self, init=None):
        if init is None:
            init = default_config
        object.__setattr__(self, "_params", dict())
        self.update(init)

    def __getitem__(self, key):
        return self._params[key]

    def __setitem__(self, key, val):
        self._params[key] = val

    def __getattr__(self, key):
        return self._params[key]

    def get(self):
        return self._params

    def update(self, init):
        for key in init:
            self[key] = init[key]
