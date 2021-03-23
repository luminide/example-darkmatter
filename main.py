import argparse
import os
import logging
import random
from glob import glob
from datetime import datetime
import cv2
import numpy as np
import imgaug

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision

from augment import make_augmenters
from report import plot_images
import hp

logging.basicConfig(
    level=logging.INFO,
    format=('%(asctime)s - %(name)s - ' '%(levelname)s -  %(message)s'))
logger = logging.getLogger('darkmatter')
parser = argparse.ArgumentParser(description='Pattern recognition')
parser.add_argument(
    '-b', '--batch-size', default=32, type=int, help='mini-batch size')
parser.add_argument(
    '-j', '--workers', default=4, type=int, metavar='N',
    help='number of data loading workers')
parser.add_argument(
    '--epochs', default=200, type=int, metavar='N',
    help='number of total epochs to run')
parser.add_argument(
    '--seed', default=None, type=int,
    help='seed for initializing the random number generator')
parser.add_argument(
    '--input', default='../input', metavar='DIR', help='input directory')

if torch.cuda.is_available():
    device = torch.device('cuda')
    img_size = 256
    logger.info('Running on GPU')
else:
    device = torch.device('cpu')
    img_size = 64
    logger.warning(
        'Running on CPU. Images will be resized to %dx%d. '
        'This will affect classification accuracy.', img_size, img_size)

class SkyDataset(data.Dataset):

    def __init__(self, input_dir, set_name, transform=None):
        self.set_name = set_name
        self.transform = transform
        file_pattern = os.path.join(
            input_dir, self.set_name, '*.png')
        self.files = sorted(glob(file_pattern))
        self.len = len(self.files)
        labels_file = os.path.join(
            input_dir, set_name + '-labels.csv')
        class_labels = np.loadtxt(
            labels_file, delimiter=',', skiprows=1, dtype=int)
        self.labels = class_labels[:, 1]

    def __getitem__(self, index):
        img = cv2.imread(self.files[index])
        img = cv2.resize(img, (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform.augment_image(img)
        img = img.transpose(2, 0, 1).copy()
        img = torch.from_numpy(img).float()
        return img, self.labels[index]

    def __len__(self):
        return self.len

def train(loader, model, criterion, optimizer):
    model.train()
    for (images, labels) in loader:
        images = images.to(device)
        labels = labels.to(device)
        # compute output
        output = model.head(model(images))
        loss = criterion(output, labels)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss

def validate(loader, model):
    model.eval()
    with torch.no_grad():
        correct = 0
        for (images, labels) in loader:
            images = images.to(device)
            labels = labels.to(device)
            output = model.head(model(images))
            pred = output.argmax(axis=1)
            correct += (pred == labels).sum()

        acc = correct*100./len(loader.dataset)
    return acc

def worker_init_fn(worker_id):
    imgaug.seed(random.randint(0, 2**32) + worker_id)

def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    input_dir = args.input
    model = torchvision.models.__dict__[hp.arch](pretrained=hp.pretrained)
    # add a new output layer
    model.head = torch.nn.Linear(1000, 2)
    model = model.to(device)

    # define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), hp.lr)

    # data loading code
    train_aug, test_aug = make_augmenters()
    train_dataset = SkyDataset(input_dir, 'training', train_aug)
    train_loader = data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        num_workers=args.workers, worker_init_fn=worker_init_fn)

    val_dataset = SkyDataset(input_dir, 'validation', test_aug)
    val_loader = data.DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=args.workers)

    writer = SummaryWriter()
    for epoch in range(args.epochs):
        # train for one epoch
        loss = train(train_loader, model, criterion, optimizer)
        writer.add_scalar('training loss', loss.item(), epoch)

        # evaluate on validation set
        acc = validate(val_loader, model)
        writer.add_scalar('validation accuracy', acc, epoch)
        print(f'Epoch {epoch + 1}: training loss {loss.item():.4f}'
              f' validation accuracy {acc:.2f}%')

    state = {
        'epoch': epoch, 'model': model.state_dict(),
        'optimizer' : optimizer.state_dict()
    }
    torch.save(state, 'model.pth')

    # get some validation images and try inference
    dataiter = iter(val_loader)
    images, labels = dataiter.next()
    with torch.no_grad():
        output = model.head(model(images.to(device)))
    output = output.cpu()
    fig = plot_images(images, labels, output, input_dir)
    writer.add_figure('Predictions vs. Actuals', fig)
    writer.close()

if __name__ == '__main__':
    main()
