import os
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

def plot_images(images, labels, output, input_dir):
    num_images = min(12, images.shape[0])
    images = images[:num_images]
    labels = labels[:num_images]
    output = output[:num_images]
    num_cols = min(4, num_images)
    num_rows = (num_images + num_cols - 1) // num_cols
    preds = output.argmax(axis=1)
    probs = [100*F.softmax(el, dim=0)[i] for i, el in zip(preds, output)]
    lenses = pd.read_csv(os.path.join(input_dir, 'test-lenses.csv'))

    # plot the images, along with predicted and true labels
    fig = plt.figure(figsize=(10, num_images))
    for idx, (img, pred, prob, label) in enumerate(
            zip(images, preds, probs, labels)):
        ax = fig.add_subplot(num_rows, num_cols, idx+1, xticks=[], yticks=[])
        npimg = np.uint8(img.numpy().round())
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        # mark the center of the lens objects
        lens = lenses[lenses['image_id'] == idx]
        if lens.shape[0] != 0:
            y = lens['row'].iloc[0]*img.shape[1]/512
            x = lens['column'].iloc[0]*img.shape[2]/512
            plt.plot(x, y, 'o', color='red', alpha=0.6)
        color = 'green' if pred == label else 'red'
        ax.set_title(
            f'pred: {pred}, {prob:.1f}%\n(label: {label})', color=color)
    plt.savefig('validation-samples.png', dpi=150)
    return fig
