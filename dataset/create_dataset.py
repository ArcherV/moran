#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import lmdb  # install lmdb by "pip install lmdb"
import cv2
import numpy as np
import tools.dataset as dataset
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import torch


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    try:
        imageBuf = np.fromstring(imageBin, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
        imgH, imgW = img.shape[0], img.shape[1]
        if imgH * imgW == 0:
            return False
    except:
        print("wrong!")
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


# outputPath:输出路径
# imagePathList:图片路径List
# labelList:标签List
def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """

    # imagePathList 对应和 labelList 配套
    assert (len(imagePathList) == len(labelList))

    # 获取数量
    nSamples = len(imagePathList)

    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]

        # 判断是否存在路径
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        # 直接读取图片
        with open(imagePath, 'r') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin

        cache[labelKey] = label

        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


def read_lmdb():
    # 读取IMDB格式的数据，并调整大小
    train_nips_dataset = dataset.lmdbDataset(root='./temp_output')
    return train_nips_dataset


if __name__ == '__main__':
    result1 = list()
    result2 = list()
    with open('./ArTtrain.txt', 'r', encoding='utf-8') as file:
        for line in file:
            item = line.strip().split(' ', 1)
            result1.append(item[0])
            result2.append(item[1])
    # print(result1[:100])
    # print(result2[:100])
    createDataset('./temp_output', result1, result2)
