import argparse
import io
import os
import random
import shutil
import sys
import time
import warnings

import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

import tools.dataset as dataset
import tools.utils as utils
from models.moran import MORAN
from tools.logger import logger
from tools.utils import showAttention
from wordlist import result

warnings.filterwarnings('ignore')

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
parser = argparse.ArgumentParser()
parser.add_argument('--train_nips', default='dataset/ArTtrain', help='path to dataset')
parser.add_argument('--train_cvpr', default='dataset/ArTtrain', help='path to dataset')
parser.add_argument('--valroot', default='dataset/ArTval', help='path to dataset')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imgH', type=int, default=64, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=200, help='the width of the input image to network')
parser.add_argument('--targetH', type=int, default=32, help='the width of the input image to network')
parser.add_argument('--targetW', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--niter', type=int, default=3000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1, help='learning rate for Critic, default=0.00005')
parser.add_argument('--cuda', action='store_false', help='enables cuda')
parser.add_argument('--MORAN', default='model_best.pth', help="path to model (to continue training)")
parser.add_argument('--alphabet', type=str,
                    default='0:1:2:3:4:5:6:7:8:9:a:b:c:d:e:f:g:h:i:j:k:l:m:n:o:p:q:r:s:t:u:v:w:x:y:z:$')
parser.add_argument('--sep', type=str, default=':')
parser.add_argument('--displayInterval', type=int, default=100, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=1000, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=40000, help='Interval to be displayed')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--adadelta', action='store_false', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--sgd', action='store_true', help='Whether to use sgd (default is rmsprop)')
opt = parser.parse_args()
print(opt)

# Modify
opt.alphabet = result

opt.manualSeed = random.randint(1, 10000)  # fix seed
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

log = logger('.')

if not torch.cuda.is_available():
    assert not opt.cuda, 'You don\'t have a CUDA device.'

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

train_nips_dataset = dataset.lmdbDataset(root=opt.train_nips,
                                         transform=dataset.resizeNormalize((opt.imgW, opt.imgH)))
assert train_nips_dataset
'''
train_cvpr_dataset = dataset.lmdbDataset(root=opt.train_cvpr, 
    transform=dataset.resizeNormalize((opt.imgW, opt.imgH)))
assert train_cvpr_dataset
'''
'''
train_dataset = torch.utils.data.ConcatDataset([train_nips_dataset, train_cvpr_dataset])
'''
train_dataset = train_nips_dataset

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    shuffle=False, sampler=dataset.randomSequentialSampler(train_dataset, opt.batchSize),
    num_workers=int(opt.workers))

test_dataset = dataset.lmdbDataset(root=opt.valroot,
                                   transform=dataset.resizeNormalize((opt.imgW, opt.imgH)))
nclass = len(opt.alphabet.split(opt.sep))
# nc = 1
nc = 3

converter = utils.strLabelConverterForAttention(opt.alphabet, opt.sep)
criterion = torch.nn.CrossEntropyLoss()

MORAN = MORAN(nc, nclass, opt.nh, opt.targetH, opt.targetW,
              inputDataType='torch.cuda.FloatTensor' if opt.cuda else 'torch.FloatTensor', CUDA=opt.cuda, log=log)

image = torch.FloatTensor(opt.batchSize, nc, opt.imgH, opt.imgW)
text = torch.LongTensor(opt.batchSize * 5)
text_rev = torch.LongTensor(opt.batchSize * 5)
length = torch.IntTensor(opt.batchSize)

if opt.cuda:
    MORAN.cuda()
    image = image.cuda()
    text = text.cuda()
    text_rev = text_rev.cuda()
    criterion = criterion.cuda()

# loss averager
loss_avg = utils.averager()

# setup optimizer
if opt.adam:
    optimizer = optim.Adam(MORAN.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
elif opt.adadelta:
    optimizer = optim.Adadelta(MORAN.parameters(), lr=opt.lr)
elif opt.sgd:
    optimizer = optim.SGD(MORAN.parameters(), lr=opt.lr, momentum=0.9)
else:
    optimizer = optim.RMSprop(MORAN.parameters(), lr=opt.lr)

if os.path.isfile(opt.MORAN):
    print("=> loading checkpoint '{}'".format(opt.MORAN))
    checkpoint = torch.load(opt.MORAN)
    opt.start_epoch = checkpoint['epoch']
    best_pred = checkpoint['best_pred']
    MORAN.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(opt.MORAN, checkpoint['epoch']))


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
        dataset, shuffle=False, batch_size=opt.batchSize, num_workers=int(opt.workers))  # opt.batchSize
    val_iter = iter(data_loader)
    max_iter = min(max_iter, len(data_loader))
    n_correct = 0
    n_total = 0
    distance = 0.0
    loss_avg = utils.averager()

    f = open('./log.txt', 'w', encoding='utf-8')

    for i in range(max_iter):
        data = val_iter.next()
        cpu_images, cpu_texts, cpu_texts_rev = data
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts, scanned=True)
        t_rev, _ = converter.encode(cpu_texts_rev, scanned=True)
        utils.loadData(text, t)
        utils.loadData(text_rev, t_rev)
        utils.loadData(length, l)
        preds0, alpha0, preds1, alpha1 = MORAN(image, length, text, text_rev, debug=True, test=True, steps=steps)
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
        alpha = []
        alpha0 = alpha0.detach().cpu().numpy()
        alpha1 = alpha1.detach().cpu().numpy()
        for j in range(cpu_images.size(0)):
            text_begin = 0 if j == 0 else length.data[:j].sum()
            if torch.mean(preds0_prob[text_begin:text_begin + len(sim_preds0[j].split('$')[0] + '$')]).item() > \
                    torch.mean(preds1_prob[text_begin:text_begin + len(sim_preds1[j].split('$')[0] + '$')]).item():
                sim_preds.append(sim_preds0[j].split('$')[0] + '$')
                alpha.append(alpha0[:, :, j])
            else:
                sim_preds.append(sim_preds1[j].split('$')[0][-1::-1] + '$')
                alpha.append(alpha1[-1::-1, :, j])

        img_shape = cpu_images.shape[3] / 100, cpu_images.shape[2] / 100
        input_seq = cpu_texts[0]
        output_seq = sim_preds[0]
        attention = alpha[0]
        attention_image = showAttention(input_seq, output_seq, attention, img_shape)
        log.image_summary('map/attention', [attention_image], steps)

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


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        print("\tSaving new best model")
        shutil.copyfile(filename, 'model_best.pth')


if __name__ == '__main__':
    t0 = time.time()
    acc, acc_tmp = 0, 0
    for epoch in range(opt.niter):
        print('Begin epoch %d' % epoch)
        train_iter = iter(train_loader)
        i = 0
        while i < len(train_loader):
            # print("main函数里,可迭代次数为 %d" %  len(train_loader))
            steps = i + epoch * len(train_loader)
            if steps % opt.valInterval == 0:
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

            if steps % opt.displayInterval == 0:
                log.scalar_summary('train loss', loss_avg.val(), steps)
                log.scalar_summary('speed', steps / (time.time() - t0), steps)
                loss_avg.reset()

            i += 1
