import argparse
import os
import random
import io
import sys
import time
from models.moran import MORAN
import tools.utils as utils
import torch.optim as optim
import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data
import tools.dataset as dataset
from wordlist import result
from torch.autograd import Variable
from collections import OrderedDict
from tools.logger import logger
import warnings
warnings.filterwarnings('ignore')

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
parser = argparse.ArgumentParser()
parser.add_argument('--train_nips', default='dataset/ArTtrain', help='path to dataset')
parser.add_argument('--train_cvpr', default='dataset/ArTtrain', help='path to dataset')
parser.add_argument('--valroot', default='dataset/ArTtest', help='path to dataset')
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
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--MORAN', default='', help="path to model (to continue training)")
parser.add_argument('--alphabet', type=str,
                    default='0:1:2:3:4:5:6:7:8:9:a:b:c:d:e:f:g:h:i:j:k:l:m:n:o:p:q:r:s:t:u:v:w:x:y:z:$')
parser.add_argument('--sep', type=str, default=':')
parser.add_argument('--experiment', default='output', help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=100, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=1000, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=40000, help='Interval to be displayed')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--adadelta', action='store_false', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--sgd', action='store_true', help='Whether to use sgd (default is rmsprop)')
parser.add_argument('--BidirDecoder', action='store_false', help='Whether to use BidirDecoder')
opt = parser.parse_args()
print(opt)

# Modify
opt.alphabet = result

assert opt.ngpu == 1, "Multi-GPU training is not supported yet, due to the variant lengths of the text in a batch."

if opt.experiment is None:
    opt.experiment = 'expr'
os.system('mkdir {0}'.format(opt.experiment))

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
                                         transform=dataset.resizeNormalize((opt.imgW, opt.imgH)),
                                         reverse=opt.BidirDecoder)
assert train_nips_dataset
'''
train_cvpr_dataset = dataset.lmdbDataset(root=opt.train_cvpr, 
    transform=dataset.resizeNormalize((opt.imgW, opt.imgH)), reverse=opt.BidirDecoder)
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
                                   transform=dataset.resizeNormalize((opt.imgW, opt.imgH)), reverse=opt.BidirDecoder)
nclass = len(opt.alphabet.split(opt.sep))
# nc = 1
nc = 3

converter = utils.strLabelConverterForAttention(opt.alphabet, opt.sep)
criterion = torch.nn.CrossEntropyLoss()

if opt.cuda:
    MORAN = MORAN(nc, nclass, opt.nh, opt.targetH, opt.targetW, BidirDecoder=opt.BidirDecoder, CUDA=opt.cuda, log=log)
else:
    MORAN = MORAN(nc, nclass, opt.nh, opt.targetH, opt.targetW, BidirDecoder=opt.BidirDecoder,
                  inputDataType='torch.FloatTensor', CUDA=opt.cuda, log=log)

if opt.MORAN != '':
    print('loading pretrained model from %s' % opt.MORAN)
    if opt.cuda:
        state_dict = torch.load(opt.MORAN)
    else:
        state_dict = torch.load(opt.MORAN, map_location='cpu')
    MORAN_state_dict_rename = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")  # remove `module.`
        MORAN_state_dict_rename[name] = v
    MORAN.load_state_dict(MORAN_state_dict_rename, strict=True)

image = torch.FloatTensor(opt.batchSize, nc, opt.imgH, opt.imgW)
text = torch.LongTensor(opt.batchSize * 5)
text_rev = torch.LongTensor(opt.batchSize * 5)
length = torch.IntTensor(opt.batchSize)

if opt.cuda:
    MORAN.cuda()
    MORAN = torch.nn.DataParallel(MORAN, device_ids=range(opt.ngpu))
    image = image.cuda()
    text = text.cuda()
    text_rev = text_rev.cuda()
    criterion = criterion.cuda()

image = Variable(image)
text = Variable(text)
text_rev = Variable(text_rev)
length = Variable(length)

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

    f = open('./log.txt', 'a', encoding='utf-8')

    for i in range(max_iter):
        data = val_iter.next()
        if opt.BidirDecoder:
            cpu_images, cpu_texts, cpu_texts_rev = data
            utils.loadData(image, cpu_images)
            t, l = converter.encode(cpu_texts, scanned=True)
            t_rev, _ = converter.encode(cpu_texts_rev, scanned=True)
            utils.loadData(text, t)
            utils.loadData(text_rev, t_rev)
            utils.loadData(length, l)
            preds0, preds1 = MORAN(image, length, text, text_rev, debug=True, test=True)
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
        else:
            cpu_images, cpu_texts = data
            utils.loadData(image, cpu_images)
            t, l = converter.encode(cpu_texts, scanned=True)
            utils.loadData(text, t)
            utils.loadData(length, l)
            preds = MORAN(image, length, text, text_rev, test=True)
            cost = criterion(preds, text)
            _, preds = preds.max(1)
            preds = preds.view(-1)
            sim_preds = converter.decode(preds.data, length.data)

        loss_avg.add(cost)
        for pred, target in zip(sim_preds, cpu_texts):
            if pred == target.lower():
                n_correct += 1
            f.write("pred %s\t\t\ttarget %s\n" % (pred, target))
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
    if opt.BidirDecoder:
        cpu_images, cpu_texts, cpu_texts_rev = data
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts, scanned=True)
        t_rev, _ = converter.encode(cpu_texts_rev, scanned=True)
        utils.loadData(text, t)
        utils.loadData(text_rev, t_rev)
        utils.loadData(length, l)
        preds0, preds1 = MORAN(image, length, text, text_rev)
        cost = criterion(torch.cat([preds0, preds1], 0), torch.cat([text, text_rev], 0))
    else:
        cpu_images, cpu_texts = data
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts, scanned=True)
        utils.loadData(text, t)
        utils.loadData(length, l)
        preds = MORAN(image, length, text, text_rev)
        cost = criterion(preds, text)

    MORAN.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


if __name__ == '__main__':
    t0 = time.time()
    acc, acc_tmp = 0, 0
    for epoch in range(opt.niter):

        train_iter = iter(train_loader)
        i = 0
        while i < len(train_loader):
            # print("main函数里,可迭代次数为 %d" %  len(train_loader))
            steps = i + epoch * len(train_loader)
            if i % opt.valInterval == 0:
                MORAN.eval()

                acc_tmp = val(test_dataset, criterion, steps=steps)
                if acc_tmp > acc:
                    acc = acc_tmp
                    torch.save(MORAN.state_dict(), '{0}/{1}_{2}.pth'.format(
                        opt.experiment, i, str(acc)[:6]))

            MORAN.train()

            cost = trainBatch()
            loss_avg.add(cost)

            if i % opt.displayInterval == 0:
                log.scalar_summary('train loss', loss_avg.val(), steps)
                log.scalar_summary('speed', steps / (time.time() - t0), steps)
                loss_avg.reset()

            i += 1
