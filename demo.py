import torch
import os
import tools.dataset as dataset
import tools.utils as utils
from models.moran import MORAN
from wordlist import result
import torch.nn.functional as F
import csv
import warnings

warnings.filterwarnings('ignore')


class params:
    model_path = 'logger/model_best.pth'
    alphabet = result
    batch_size = 128
    nc = 3
    nh = 256
    imgH = 32
    imgW = 100
    num_workers = 4
    max_iter = 20
    span_range_width = .9
    span_range_height = .9
    grid_height = 4
    grid_width = 4
    image_height = 32
    image_width = 100
    model = 'unbounded_stn'
    data = 'sp-crop'


args = params()

MORAN = MORAN(args, args.nc, len(args.alphabet.split(':')), args.nh, args.imgH, args.imgW).cuda()

if os.path.isfile(args.model_path):
    model_dict = MORAN.state_dict()
    pretrained_model = torch.load(args.model_path)
    pretrained_model = {k.replace('module.', ''): v for k, v in pretrained_model.items()}
    pretrained_model = {k: v for k, v in pretrained_model.items() if k in model_dict}
    model_dict.update(pretrained_model)
    MORAN.load_state_dict(model_dict)

MORAN.eval()

converter = utils.strLabelConverterForAttention(args.alphabet, ':')
pred_dataset = dataset.lmdbDataset(root=os.path.join('dataset', args.data), transform=dataset.resizeNormalize((100, 32)))
pred_loader = torch.utils.data.DataLoader(
        pred_dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

image = torch.FloatTensor(args.batch_size, args.nc, args.imgH, args.imgW).cuda()
text = torch.LongTensor(args.batch_size * 5).cuda()
length = torch.IntTensor(args.batch_size).cuda()
t, l = converter.encode(['0' * args.max_iter] * args.batch_size, scanned=True)
utils.loadData(text, t)
utils.loadData(length, l)

f = open(os.path.join('logger', args.data + '.csv'), 'w', newline='', encoding='utf-8')
writer = csv.writer(f)

for i, (img_keys, cpu_images) in enumerate(pred_loader):
    utils.loadData(image, cpu_images)
    t, l = converter.encode(['0' * args.max_iter] * cpu_images.size(0), scanned=True)
    utils.loadData(text, t)
    utils.loadData(length, l)
    preds0, _, preds1, _ = MORAN(image, length, text, text, test=True)
    preds0 = F.softmax(preds0, dim=1)
    preds0_prob, preds0 = preds0.max(1)
    preds0 = preds0.view(-1)
    preds0_prob = preds0_prob.view(-1)
    sim_preds0 = converter.decode(preds0.data, length.data)
    preds1 = F.softmax(preds1, dim=1)
    preds1_prob, preds1 = preds1.max(1)
    preds1 = preds1.view(-1)
    preds1_prob = preds1_prob.view(-1)
    sim_preds1 = converter.decode(preds1.data, length.data)
    sim_preds, sim_prob = [], []
    for j in range(cpu_images.size(0)):
        text_begin = 0 if j == 0 else length.data[:j].sum()
        preds0_p = torch.mean(preds0_prob[text_begin:text_begin + len(sim_preds0[j].split('$')[0] + '$')]).item()
        preds1_p = torch.mean(preds1_prob[text_begin:text_begin + len(sim_preds1[j].split('$')[0] + '$')]).item()
        if preds0_p > preds1_p:
            sim_preds.append(sim_preds0[j].split('$')[0])
            sim_prob.append(preds0_p)
        else:
            sim_preds.append(sim_preds1[j].split('$')[0][-1::-1])
            sim_prob.append(preds1_p)

    for key, probility, pred in zip(img_keys, sim_prob, sim_preds):
        writer.writerow([key, '%.2f%%' % (probility * 100), pred])
f.close()
