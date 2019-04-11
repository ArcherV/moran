import torch
import os
import tools.dataset as dataset
import tools.utils as utils
from models.moran import MORAN
from wordlist import result

model_path = 'model_best.pth'
alphabet = result
batch_size = 64
nc = 3
nh = 256
imgH = 32
imgW = 100
num_workers = 4
max_iter = 20

MORAN = MORAN(nc, len(alphabet.split(':')), nh, imgH, imgW).cuda()

if os.path.isfile(model_path):
    print("=> loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path)
    start_epoch = checkpoint['epoch']
    best_pred = checkpoint['best_pred']
    MORAN.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {} accuracy {})"
          .format(model_path, checkpoint['epoch'], best_pred))

MORAN.eval()

converter = utils.strLabelConverterForAttention(alphabet, ':')
pred_dataset = dataset.lmdbDataset(root='dataset/ArTtest', transform=dataset.resizeNormalize((100, 32)))
pred_loader = torch.utils.data.DataLoader(
        pred_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)

image = torch.FloatTensor(batch_size, nc, imgH, imgW).cuda()
text = torch.LongTensor(batch_size * 5).cuda()
length = torch.IntTensor(batch_size).cuda()
t, l = converter.encode(['0'*max_iter] * batch_size, scanned=True)
utils.loadData(text, t)
utils.loadData(length, l)

f = open('./pred.txt', 'w', encoding='utf-8')

for i, (img_keys, cpu_images, _, _) in enumerate(pred_loader):
    utils.loadData(image, cpu_images)
    t, l = converter.encode(['0' * max_iter] * cpu_images.size(0), scanned=True)
    utils.loadData(text, t)
    utils.loadData(length, l)
    preds0, _, preds1, _ = MORAN(image, length, text, text, test=True)
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

    for key, pred in zip(img_keys, sim_preds):
        f.write('%s:\t%s\n' % (key, pred))
        if i % 100 == 0:
            print('%s:\t%s' % (key, pred))
f.close()
