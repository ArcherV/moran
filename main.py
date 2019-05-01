import io
import os
import random
import shutil
import sys
import time
import warnings

import numpy as np
import torch.backends.cudnn as cudnn
import torch.argsim as argsim
import torch.utils.data
import tools.dataset as dataset
import tools.utils as utils
from models.moran import MORAN
from tools.logger import logger
from wordlist import result

warnings.filterwarnings('ignore')

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class params:
    trainroot = 'dataset/ArTtrain'
    valroot = 'dataset/ArTval'
    batchSize = 32
    imgH = 64
    imgW = 200
    targetH = 32
    targetW = 100
    nh = 256
    niter = 3000
    lr = 1
    cuda = True
    model_path = 'logger/model_best.pth'
    alphabet = ''
    sep = ':'
    displayInterval = 100
    n_test_disp = 10
    valInterval = 1000
    optimizer = 'adadelta'
    workers = 4


args = params()

# Modify
args.alphabet = result

args.manualSeed = random.randint(1, 10000)  # fix seed
random.seed(args.manualSeed)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

cudnn.benchmark = True

log = logger('logger')

train_dataset = dataset.lmdbDataset(root=args.trainroot,
                                    transform=dataset.resizeNormalize((args.imgW, args.imgH)))
'''
train_cvpr_dataset = dataset.lmdbDataset(root=args.train_cvpr, 
    transform=dataset.resizeNormalize((args.imgW, args.imgH)))
assert train_cvpr_dataset
'''
'''
train_dataset = torch.utils.data.ConcatDataset([train_nips_dataset, train_cvpr_dataset])
'''

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batchSize,
    shuffle=False, sampler=dataset.randomSequentialSampler(train_dataset, args.batchSize),
    num_workers=args.workers)

test_dataset = dataset.lmdbDataset(root=args.valroot,
                                   transform=dataset.resizeNormalize((args.imgW, args.imgH)))

nclass = len(args.alphabet.split(args.sep))
# nc = 3 + args.imgH + args.imgW
nc = 3

converter = utils.strLabelConverterForAttention(args.alphabet, args.sep)
criterion = torch.nn.CrossEntropyLoss()

MORAN = MORAN(nc, nclass, args.nh, args.targetH, args.targetW,
              CUDA=args.cuda)

image = torch.FloatTensor(args.batchSize, nc, args.imgH, args.imgW)
text = torch.LongTensor(args.batchSize * 5)
text_rev = torch.LongTensor(args.batchSize * 5)
length = torch.IntTensor(args.batchSize)

if args.cuda:
    MORAN.cuda()
    image = image.cuda()
    text = text.cuda()
    text_rev = text_rev.cuda()
    criterion = criterion.cuda()

# loss averager
loss_avg = utils.averager()

# setup optimizer
if args.optimizer == 'adam':
    optimizer = argsim.Adam(MORAN.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
elif args.optimizer == 'adadelta':
    optimizer = argsim.Adadelta(MORAN.parameters(), lr=args.lr)
elif args.optimizer == 'sgd':
    optimizer = argsim.SGD(MORAN.parameters(), lr=args.lr, momentum=0.9)
else:
    optimizer = argsim.RMSprop(MORAN.parameters(), lr=args.lr)

if os.path.isfile(args.model_path):
    print('=>loading pretrained model from %s for val only.' % args.resume)
    checkpoint = torch.load(args.model_path)
    pretrained_model = checkpoint['state_dict']
    model_dict = MORAN.state_dict()
    pretrained_model = {k: v for k, v in pretrained_model.items() if k in model_dict}
    model_dict.update(pretrained_model)
    MORAN.load_state_dict(model_dict)
    optimizer.load_state_dict(checkpoint['optimizer'])
    args.start_epoch = checkpoint['epoch'] + 1
    print("=> loaded checkpoint '{}' (epoch {}) (accuracy {})".format(args.model_path, checkpoint['epoch'],
                                                                      checkpoint['best_pred']))
else:
    print('Training from scratch!')


def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def val(dataset, criterion, max_iter=10000, steps=0):
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=False, batch_size=args.batchSize, num_workers=args.workers)  # args.batchSize
    val_iter = iter(data_loader)
    max_iter = min(max_iter, len(data_loader))
    n_correct = 0
    n_total = 0
    distance = 0.0
    loss_avg = utils.averager()

    f = open('logger/log.txt', 'w', encoding='utf-8')

    for i in range(max_iter):
        data = val_iter.next()
        cpu_images, cpu_texts, cpu_texts_rev = data
        # utils.loadData(image, encode_coordinates_fn(cpu_images))
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts, scanned=True)
        t_rev, _ = converter.encode(cpu_texts_rev, scanned=True)
        utils.loadData(text, t)
        utils.loadData(text_rev, t_rev)
        utils.loadData(length, l)
        preds0, _, preds1, _ = MORAN(image, length, text, text_rev, debug=False, test=True, steps=steps)
        cost = criterion(torch.cat([preds0, preds1], 0), torch.cat([text, text_rev], 0))
        preds0_prob, preds0 = preds0.max(1)
        preds0 = preds0.view(-1)
        preds0_prob = preds0_prob.view(-1)
        sim_preds0 = converter.decode(preds0.data, length.data)
        preds1_prob, preds1 = preds1.max(1)
        preds1 = preds1.view(-1)
        preds1_prob = preds1_prob.view(-1)
        sim_preds1 = converter.decode(preds1.data, length.data)
        sim_preds = []
        for j in range(cpu_images.size(0)):
            text_begin = 0 if j == 0 else length.data[:j].sum()
            if torch.mean(preds0_prob[text_begin:text_begin + len(sim_preds0[j].split('$')[0] + '$')]).item() > \
                    torch.mean(preds1_prob[text_begin:text_begin + len(sim_preds1[j].split('$')[0] + '$')]).item():
                sim_preds.append(sim_preds0[j].split('$')[0] + '$')
            else:
                sim_preds.append(sim_preds1[j].split('$')[0][-1::-1] + '$')

        # img_shape = cpu_images.shape[3] / 100, cpu_images.shape[2] / 100
        # input_seq = cpu_texts[0]
        # output_seq = sim_preds[0]
        # attention = alpha[0]
        # attention_image = showAttention(input_seq, output_seq, attention, img_shape)
        # log.image_summary('map/attention', [attention_image], steps)

        loss_avg.add(cost)
        for pred, target in zip(sim_preds, cpu_texts):
            if pred == target.lower():
                n_correct += 1
            f.write("pred %s\t\t\t\t\ttarget %s\n" % (pred, target))
            distance += levenshtein(pred, target) / max(len(pred), len(target))
            n_total += 1

    f.close()

    accuracy = n_correct / float(n_total)
    log.scalar_summary('Validation/levenshtein distance', distance / n_total, steps)
    log.scalar_summary('Validation/loss', loss_avg.val(), steps)
    log.scalar_summary('Validation/accuracy', accuracy, steps)
    return accuracy


def trainBatch():
    data = train_iter.next()
    cpu_images, cpu_texts, cpu_texts_rev = data
    # utils.loadData(image, encode_coordinates_fn(cpu_images))
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts, scanned=True)
    t_rev, _ = converter.encode(cpu_texts_rev, scanned=True)
    utils.loadData(text, t)
    utils.loadData(text_rev, t_rev)
    utils.loadData(length, l)
    preds0, preds1 = MORAN(image, length, text, text_rev)
    cost = criterion(torch.cat([preds0, preds1], 0), torch.cat([text, text_rev], 0))

    MORAN.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


def save_checkpoint(state, is_best, filename='logger/checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        print("\tSaving new best model")
        shutil.copyfile(filename, 'logger/model_best.pth')


if __name__ == '__main__':
    t0 = time.time()
    acc, acc_tmp = 0, 0
    for epoch in range(args.niter):
        print('Begin epoch %d' % epoch)
        train_iter = iter(train_loader)
        i = 0
        while i < len(train_loader):
            # print("main函数里,可迭代次数为 %d" %  len(train_loader))
            steps = i + epoch * len(train_loader)
            if steps % args.valInterval == 0:
                MORAN.eval()

                print('Begin validating')
                acc_tmp = val(test_dataset, criterion, steps=steps)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': MORAN.state_dict(),
                    'best_pred': acc,
                    'optimizer': optimizer.state_dict(),
                }, is_best=acc_tmp > acc)
                acc = max(acc_tmp, acc)

            MORAN.train()

            cost = trainBatch()
            loss_avg.add(cost)

            if steps % args.displayInterval == 0:
                log.scalar_summary('Train/loss', loss_avg.val(), steps)
                log.scalar_summary('Train/speed', steps / (time.time() - t0), steps)
                loss_avg.reset()

            i += 1
