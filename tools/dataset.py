import random
import sys
import numpy as np
import lmdb
import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import sampler

from tools.words import word


class lmdbDataset(Dataset):

    def __init__(self, root=None, transform=None, alphabet=word):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % root)
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

        self.transform = transform
        self.alphabet = alphabet

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        # print("在tools.dataset里,",index)
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index  # For training
            # img_key = 'gt_%d' % index
            imageBuf = np.fromstring(txn.get(img_key.encode()), dtype=np.uint8)
            img = cv2.imdecode(imageBuf, cv2.IMREAD_COLOR)

            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()).decode('utf-8'))
            # label = '1'

            label = ''.join(label[i] if label[i].lower() in self.alphabet else ''
                            for i in range(len(label)))
            # print("在tools.dataset里,label为",label)
            if len(label) <= 0:
                return self[index + 1]
            label_rev = label[-1::-1]
            label_rev += '$'
            label += '$'

            if self.transform:
                img = self.transform(img)

        return img, label, label_rev  # For training
        # return img_key, img, label, label_rev


class resizeNormalize(object):

    def __init__(self, size):
        self.size = size
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = cv2.resize(img, self.size)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.arange(0, self.batch_size)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.arange(0, tail)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)


def encode_coordinates_fn(net):
    batch, _, h, w = net.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x= torch.from_numpy(x.reshape(1, h, w))
    y= torch.from_numpy(y.reshape(1, h, w))
    w_loc = torch.FloatTensor(w, h, w).zero_()
    h_loc = torch.FloatTensor(h, h, w).zero_()
    w_loc.scatter_(dim=0, index=x, value=1)
    h_loc.scatter_(dim=0, index=y, value=1)
    loc = torch.cat([h_loc, w_loc], 0)
    loc = loc.expand(batch, h + w, h, w)
    net = torch.cat([net, loc], 1)
    return net
