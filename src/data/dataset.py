#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import numpy as np
import os
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.utils.data import Dataset

from src.utils.utils import truncate_seq_pair, numpy_seed

import random

import torchvision.transforms as transforms
from torchvision.datasets.folder import make_dataset
import sys
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision.transforms import functional as F
class JsonlDataset(Dataset):
    def __init__(self, data_path, tokenizer, transforms, vocab, args):
        self.data = [json.loads(l) for l in open(data_path)]
        # print(data_path)
        self.data_dir = os.path.dirname(data_path)
        self.tokenizer = tokenizer
        self.args = args
        self.vocab = vocab
        self.n_classes = len(args.labels)
        self.text_start_token = ["[CLS]"] if args.model != "mmbt" else ["[SEP]"]

        with numpy_seed(0):
            for row in self.data:
                if np.random.random() < args.drop_img_percent:
                    row["img"] = None

        self.max_seq_len = args.max_seq_len
        if args.model == "mmbt":
            self.max_seq_len -= args.num_image_embeds

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        if self.args.task == "vsnli":
            sent1 = self.tokenizer(self.data[index]["sentence1"])
            sent2 = self.tokenizer(self.data[index]["sentence2"])
            truncate_seq_pair(sent1, sent2, self.args.max_seq_len - 3)
            sentence = self.text_start_token + sent1 + ["[SEP]"] + sent2 + ["[SEP]"]
            segment = torch.cat(
                [torch.zeros(2 + len(sent1)), torch.ones(len(sent2) + 1)]
            )
        else:
            _ = self.tokenizer(self.data[index]["text"])
            if self.args.noise > 0.0:
                p = [0.5, 0.5]
                flag = np.random.choice([0, 1], p=p)
                if flag:
                    wordlist=self.data[index]["text"].split(' ')
                    for i in range(len(wordlist)):
                        replace_p=1/10*self.args.noise
                        # print(replace_p)
                        replace_flag = np.random.choice([0, 1], p=[1-replace_p, replace_p])
                        if replace_flag:
                            # pass
                            wordlist[i]='_'
                    _=' '.join(wordlist)
                    _=self.tokenizer(_)
                    # print(_)
                    # exit(1)
                    # print("src:",self.data[index]["text"]," replace:",_,'\n')

            sentence = (
                self.text_start_token
                + _[:(self.args.max_seq_len - 1)]
            )
            segment = torch.zeros(len(sentence))

        sentence = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                for w in sentence
            ]
        )


        if self.args.task_type == "multilabel":
            label = torch.zeros(self.n_classes)
            label[
                [self.args.labels.index(tgt) for tgt in self.data[index]["label"]]
            ] = 1
        else:
            label = torch.LongTensor(
                [self.args.labels.index(self.data[index]["label"])]
            )

        image = None
        if self.args.model in ["img", "concatbow","latefusion"]:
            if self.data[index]["img"]:
                img_path = os.path.join(self.data_dir, self.data[index]["img"])
                # print(img_path)
                if self.args.task == "vsnli":
                    img_path = os.path.join(self.data_dir,'flickr30k-images', self.data[index]["img"])
                image = Image.open(
                    img_path
                ).convert("RGB")
            else:
                image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))
            image = self.transforms(image)
            # print(image)
        if self.args.model == "mmbt":
            # The first SEP is part of Image Token.
            segment = segment[1:]
            sentence = sentence[1:]
            # The first segment (0) is of images.
            segment += 1

        # print(image)

        return sentence, segment, image, label,torch.LongTensor([index])

class AddGaussianNoise(object):

    '''
    mean:均值
    variance：方差
    amplitude：幅值
    '''
    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):

        img = np.array(img)
        h, w, c = img.shape
        np.random.seed(0)
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
        N = np.repeat(N, c, axis=2)
        img = N + img
        img[img > 255] = 255                       # 避免有值超过255而反转
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img

class AddSaltPepperNoise(object):

    def __init__(self, density=0,p=0.5):
        self.density = density
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:  # 概率的判断
            img = np.array(img)  # 图片转numpy
            h, w, c = img.shape
            Nd = self.density
            Sd = 1 - Nd
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd / 2.0, Nd / 2.0, Sd])  # 生成一个通道的mask
            mask = np.repeat(mask, c, axis=2)  # 在通道的维度复制，生成彩色的mask
            img[mask == 0] = 0  # 椒
            img[mask == 1] = 255  # 盐
            img = Image.fromarray(img.astype('uint8')).convert('RGB')  # numpy转图片
            return img
        else:
            return img
class AlignedConcDataset:

    def __init__(self, cfg, data_dir=None, transform=None, labeled=True):
        self.cfg = cfg
        self.transform = transform
        self.data_dir = data_dir
        self.labeled = labeled

        self.classes, self.class_to_idx = find_classes(self.data_dir)
        self.int_to_class = dict(zip(range(len(self.classes)), self.classes))
        self.imgs = make_dataset(self.data_dir, self.class_to_idx, 'png')

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        if self.labeled:
            img_path, label = self.imgs[index]
        else:
            img_path = self.imgs[index]

        img_name = os.path.basename(img_path)
        AB_conc = Image.open(img_path).convert('RGB')

        # split RGB and Depth as A and B
        w, h = AB_conc.size
        w2 = int(w / 2)
        if w2 > self.cfg.FINE_SIZE:
            A = AB_conc.crop((0, 0, w2, h)).resize((self.cfg.LOAD_SIZE, self.cfg.LOAD_SIZE), Image.BICUBIC)
            B = AB_conc.crop((w2, 0, w, h)).resize((self.cfg.LOAD_SIZE, self.cfg.LOAD_SIZE), Image.BICUBIC)
        else:
            A = AB_conc.crop((0, 0, w2, h))
            B = AB_conc.crop((w2, 0, w, h))

        if self.labeled:
            sample = {'A': A, 'B': B, 'img_name': img_name, 'label': label, 'idx': index}
        else:
            sample = {'A': A, 'B': B, 'img_name': img_name, 'idx': index}

        if self.transform:
            sample['A'] = self.transform(sample['A'])
            sample['B'] = self.transform(sample['B'])

        return sample


class RandomCrop(transforms.RandomCrop):

    def __call__(self, sample):
        A, B = sample['A'], sample['B']

        if self.padding > 0:
            A = F.pad(A, self.padding)
            B = F.pad(B, self.padding)

        # pad the width if needed
        if self.pad_if_needed and A.size[0] < self.size[1]:
            A = F.pad(A, (int((1 + self.size[1] - A.size[0]) / 2), 0))
            B = F.pad(B, (int((1 + self.size[1] - B.size[0]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and A.size[1] < self.size[0]:
            A = F.pad(A, (0, int((1 + self.size[0] - A.size[1]) / 2)))
            B = F.pad(B, (0, int((1 + self.size[0] - B.size[1]) / 2)))

        i, j, h, w = self.get_params(A, self.size)
        sample['A'] = F.crop(A, i, j, h, w)
        sample['B'] = F.crop(B, i, j, h, w)

        # _i, _j, _h, _w = self.get_params(A, self.size)
        # sample['A'] = F.crop(A, i, j, h, w)
        # sample['B'] = F.crop(B, _i, _j, _h, _w)

        return sample


class CenterCrop(transforms.CenterCrop):

    def __call__(self, sample):
        A, B = sample['A'], sample['B']
        sample['A'] = F.center_crop(A, self.size)
        sample['B'] = F.center_crop(B, self.size)
        return sample


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):

    def __call__(self, sample):
        A, B = sample['A'], sample['B']
        if random.random() > 0.5:
            A = F.hflip(A)
            B = F.hflip(B)

        sample['A'] = A
        sample['B'] = B

        return sample


def find_classes(dir):
    """
    Finds the class folders in a dataset.

    Args:
        dir (string): Root directory path.

    Returns:
        tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

    Ensures:
        No class is a subdirectory of another.
    """
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


class Resize(transforms.Resize):

    def __call__(self, sample):
        A, B = sample['A'], sample['B']
        h = self.size[0]
        w = self.size[1]

        sample['A'] = F.resize(A, (h, w))
        sample['B'] = F.resize(B, (h, w))

        return sample


class ToTensor(object):
    def __call__(self, sample):
        A, B = sample['A'], sample['B']

        # if isinstance(sample, dict):
        #     for key, value in sample:
        #         _list = sample[key]
        #         sample[key] = [F.to_tensor(item) for item in _list]

        sample['A'] = F.to_tensor(A)
        sample['B'] = F.to_tensor(B)

        return sample


class Normalize(transforms.Normalize):

    def __call__(self, sample):
        A, B = sample['A'], sample['B']
        sample['A'] = F.normalize(A, self.mean, self.std)
        sample['B'] = F.normalize(B, self.mean, self.std)

        return sample


class Lambda(transforms.Lambda):

    def __call__(self, sample):
        return self.lambd(sample)