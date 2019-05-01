import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from models.convnet import se_resnet50
from models.fracPickup import fracPickup


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, dropout=0.3)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class AttentionCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_embeddings=128, CUDA=True):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.GRUCell(input_size + num_embeddings, hidden_size)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_embeddings = num_embeddings
        self.fracPickup = fracPickup(CUDA=CUDA)

    def forward(self, prev_hidden, feats, cur_embeddings, test=False):
        nT = feats.size(0)
        nB = feats.size(1)
        nC = feats.size(2)
        hidden_size = self.hidden_size

        feats_proj = self.i2h(feats.view(-1, nC))
        prev_hidden_proj = self.h2h(prev_hidden).view(1, nB, hidden_size).expand(nT, nB, hidden_size).contiguous().view(
            -1, hidden_size)
        emition = self.score(torch.tanh(feats_proj + prev_hidden_proj).view(-1, hidden_size)).view(nT, nB)

        alpha = F.softmax(emition, 0)  # nB * nT

        if not test:
            alpha_fp = self.fracPickup(alpha.unsqueeze(1).unsqueeze(2)).squeeze()
            context = (feats * alpha_fp.view(nT, nB, 1).expand(nT, nB, nC)).sum(0).squeeze(0)  # nB * nC
            if len(context.size()) == 1:
                context = context.unsqueeze(0)
            context = torch.cat([context, cur_embeddings], 1)
            cur_hidden = self.rnn(context, prev_hidden)
            return cur_hidden, alpha_fp
        else:
            context = (feats * alpha.view(nT, nB, 1).expand(nT, nB, nC)).sum(0).squeeze(0)  # nB * nC
            if len(context.size()) == 1:
                context = context.unsqueeze(0)
            context = torch.cat([context, cur_embeddings], 1)
            cur_hidden = self.rnn(context, prev_hidden)
            return cur_hidden, alpha


class Attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_embeddings=128, CUDA=True):
        super(Attention, self).__init__()
        self.attention_cell = AttentionCell(input_size, hidden_size, num_embeddings, CUDA=CUDA)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.generator = nn.Linear(hidden_size, num_classes)
        self.char_embeddings = Parameter(torch.randn(num_classes + 1, num_embeddings))
        self.num_embeddings = num_embeddings
        self.num_classes = num_classes
        self.cuda = CUDA

    # targets is nT * nB
    def forward(self, feats, text_length, text, test=False):

        nT = feats.size(0)
        nB = feats.size(1)
        nC = feats.size(2)
        hidden_size = self.hidden_size
        input_size = self.input_size
        assert (input_size == nC)
        assert (nB == text_length.numel())

        num_steps = text_length.data.max()
        num_labels = text_length.data.sum()

        if not test:

            targets = torch.zeros(nB, num_steps + 1).long()
            if self.cuda:
                targets = targets.cuda()
            start_id = 0

            for i in range(nB):
                targets[i][1:1 + text_length.data[i]] = text.data[start_id:start_id + text_length.data[i]] + 1
                start_id = start_id + text_length.data[i]
            targets = Variable(targets.transpose(0, 1).contiguous())

            output_hiddens = Variable(torch.zeros(num_steps, nB, hidden_size).type_as(feats.data))
            hidden = Variable(torch.zeros(nB, hidden_size).type_as(feats.data))

            for i in range(num_steps):
                cur_embeddings = self.char_embeddings.index_select(0, targets[i])
                hidden, alpha = self.attention_cell(hidden, feats, cur_embeddings, test)
                output_hiddens[i] = hidden

            new_hiddens = Variable(torch.zeros(num_labels, hidden_size).type_as(feats.data))
            b = 0
            start = 0

            for length in text_length.data:
                new_hiddens[start:start + length] = output_hiddens[0:length, b, :]
                start = start + length
                b = b + 1

            probs = self.generator(new_hiddens)
            return probs

        else:

            hidden = Variable(torch.zeros(nB, hidden_size).type_as(feats.data))
            targets_temp = Variable(torch.zeros(nB).long().contiguous())
            probs = Variable(torch.zeros(nB * num_steps, self.num_classes))
            output_alpha = Variable(torch.zeros(num_steps, nT, nB)).type_as(feats.data)
            if self.cuda:
                targets_temp = targets_temp.cuda()
                probs = probs.cuda()

            for i in range(num_steps):
                cur_embeddings = self.char_embeddings.index_select(0, targets_temp)
                hidden, alpha = self.attention_cell(hidden, feats, cur_embeddings, test)
                output_alpha[i] = alpha
                hidden2class = self.generator(hidden)
                probs[i * nB:(i + 1) * nB] = hidden2class
                _, targets_temp = hidden2class.max(1)
                targets_temp += 1

            probs = probs.view(num_steps, nB, self.num_classes).permute(1, 0, 2).contiguous()
            probs = probs.view(-1, self.num_classes).contiguous()
            probs_res = Variable(torch.zeros(num_labels, self.num_classes).type_as(feats.data))
            b = 0
            start = 0

            for length in text_length.data:
                probs_res[start:start + length] = probs[b * num_steps:b * num_steps + length]
                start = start + length
                b += 1

            return probs_res, output_alpha


class ASRN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, CUDA=True):
        super(ASRN, self).__init__()
        assert imgH % 16 == 0, 'imgH must be a multiple of 16'

        self.se_resnet50 = se_resnet50()

        self.rnn = nn.Sequential(
            BidirectionalLSTM(2048, nh, nh),
            BidirectionalLSTM(nh, nh, nh),
        )

        self.attentionL2R = Attention(nh, nh, nclass, 256, CUDA=CUDA)
        self.attentionR2L = Attention(nh, nh, nclass, 256, CUDA=CUDA)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', a=0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input, length, text, text_rev, test=False):
        # conv features
        conv = self.se_resnet50(input)

        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1).contiguous()  # [w, b, c]

        # rnn features
        rnn = self.rnn(conv)

        if test:
            outputL2R, alpha0 = self.attentionL2R(rnn, length, text, test)
            outputR2L, alpha1 = self.attentionR2L(rnn, length, text_rev, test)
            return outputL2R, alpha0, outputR2L, alpha1
        else:
            outputL2R = self.attentionL2R(rnn, length, text, test)
            outputR2L = self.attentionR2L(rnn, length, text_rev, test)
            return outputL2R, outputR2L


if __name__ == '__main__':
    model = ASRN(imgH=32, nc=3, nclass=10, nh=128).cuda()
    inp = torch.Tensor(1, 3, 32, 100).cuda()
    out = model(inp, length=10, text=None, text_rev=None)
    print(out.shape)
