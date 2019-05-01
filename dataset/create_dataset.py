#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import cv2
import lmdb  # install lmdb by "pip install lmdb"
import numpy as np


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
def createDataset(outputPath, root, imagePathList, checkValid=True):
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
    # assert (len(imagePathList) == len(labelList))

    # 获取数量
    nSamples = len(imagePathList)

    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = imagePathList[i]

        # 判断是否存在路径
        if not os.path.exists(os.path.join(root, imagePath)):
            print('%s does not exist' % imagePath)
            continue
        # 直接读取图片
        with open(os.path.join(root, imagePath), 'r') as f:
            imageBin = f.read()
        # if checkValid:
        #     if not checkImageIsValid(imageBin):
        #         print('%s is not a valid image' % imagePath)
        #         continue

        imageKey = imagePath.split('.')[0]
        imageKey = '_'.join(imageKey.split('/'))
        cache[imageKey] = imageBin

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    # imagePathList, labelList = [], []
    # for file_name in ['ArTtrain-sp.txt', 'LSVTtrain-sp.txt']:
    #     with open(file_name, 'r') as file:
    #         for line in file:
    #             imagePathList.append(line.split(' ', 1)[0])
                # labelList.append(line.split(' ', 1)[1])
    # imagePathList = os.listdir('ArTtest_final_ori')
    # createDataset('ArTtest_final', 'ArTtest_final_ori', imagePathList)
    # root = 'left'
    root = 'sp-crop'
    img_list = os.listdir(root)
    # image_list = list()
    # for img in img_list:
    #     char = os.listdir(os.path.join(root, img))
    #     for image in char:
    #         image_list.append(os.path.join(img, image))
    createDataset(root, root, img_list)

    # env = lmdb.open(
    #     'dataset/LSVT-hp',
    #     max_readers=1,
    #     readonly=True,
    #     lock=False,
    #     readahead=False,
    #     meminit=False)
    # with env.begin(write=False) as txn:
    #     for key, _ in txn.cursor():
    #         print(key)
    # file = list()
    # with env.begin(write=False) as txn:
    #     for key, _ in txn.cursor():
    #         if b'gt' in key:
    #             file.append(key.decode())
    # file = sorted(file, key=lambda x: (int(x.split('_')[1].split('/')[0]), int(x.split('/')[-1])))
    # with open('data.txt', 'w') as f:
    #     for x in file:
    #         f.write(x + '\n')
