# encoding: utf-8
import torch
import itertools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNN(nn.Module):
    def __init__(self, num_output):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(320, 50)
        self.fc1 = nn.Linear(2200, 50)
        self.fc2 = nn.Linear(50, num_output)

    def forward(self, x):
        x = x + 1
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 2200)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class ClsNet(nn.Module):

    def __init__(self):
        super(ClsNet, self).__init__()
        # self.cnn = CNN(10)###################
        self.cnn = CNN(64)

    def forward(self, x):
        # self.x=x
        # self.cnn = CNN(xx.size()[0])
        return F.log_softmax(self.cnn(x))


class BoundedGridLocNet(nn.Module):

    def __init__(self, grid_height, grid_width, target_control_points):
        super(BoundedGridLocNet, self).__init__()
        self.cnn = CNN(grid_height * grid_width * 2)

        bias = torch.from_numpy(np.arctanh(target_control_points.numpy()))
        bias = bias.view(-1)
        self.cnn.fc2.bias.data.copy_(bias)
        self.cnn.fc2.weight.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        points = F.tanh(self.cnn(x))
        return points.view(batch_size, -1, 2)


class UnBoundedGridLocNet(nn.Module):

    def __init__(self, grid_height, grid_width, target_control_points):
        super(UnBoundedGridLocNet, self).__init__()
        # print('grid_height',grid_height) #4
        # print('grid_width', grid_width) #4
        # print('grid_height * grid_width * 2',grid_height * grid_width * 2) #32
        self.cnn = CNN(grid_height * grid_width * 2)

        bias = target_control_points.view(-1)
        # print('bias',bias.size()) #([32])
        self.cnn.fc2.bias.data.copy_(bias)
        # self.cnn.fc2.weight.data=self.cnn.fc2.weight.data/10
        # self.cnn.fc2.weight.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        # print('batch_size', batch_size) #64 #64
        points = self.cnn(x)
        # print('x',x.size()) #([64,1,28,28]) #([64,1,30,100])
        # print('points',points.size()) #([64,32]) #([352,32])
        t = points.view(batch_size, -1, 2)
        # print('t',t.size()) #([64,16,2]) #([64,88,2])
        return points.view(batch_size, -1, 2)


class STNClsNet(nn.Module):

    def __init__(self, opt):
        super(STNClsNet, self).__init__()
        self.opt = opt

        r1 = opt.span_range_height  # 0.9
        r2 = opt.span_range_width  # 0.9
        # r2 = 0.9
        assert r1 < 1 and r2 < 1  # if >= 1, arctanh will cause error in BoundedGridLocNet
        # print('args.grid_height',args.grid_height) #4
        # print('args.grid_width', args.grid_width) #4

        target_control_points = torch.Tensor(list(itertools.product(
            np.arange(-r1, r1 + 0.00001, 2.0 * r1 / (opt.grid_height - 1)),
            np.arange(-r2, r2 + 0.00001, 2.0 * r2 / (opt.grid_width - 1)),
        )))

        # print('target_control_points', target_control_points) #([16,2])
        Y, X = target_control_points.split(1, dim=1)
        # print('Y', Y.size()) ([16,1])
        # print('X', X.size()) ([16,1])
        target_control_points = torch.cat([X, Y], dim=1)
        # print('target_control_points1', target_control_points) #([16,2])
        GridLocNet = {
            'unbounded_stn': UnBoundedGridLocNet,
            'bounded_stn': BoundedGridLocNet,
        }[opt.model]
        self.loc_net = GridLocNet(opt.grid_height, opt.grid_width, target_control_points)

        self.tps = TPSGridGen(opt.image_height, opt.image_width, target_control_points)

        # self.cls_net = ClsNet()

    def forward(self, x):
        # print('forward(self, x)', x.size()) #([64,1,28,28]) #([64,1,30,100])
        batch_size = x.size(0)
        source_control_points = self.loc_net(x)
        # print('source_control_points',source_control_points) #([64,16,2])  #([64,88,2])
        source_coordinate = self.tps(source_control_points)
        # print('source_coordinate', source_coordinate.size()) #([64,784,2])
        grid = source_coordinate.view(batch_size, self.opt.image_height, self.opt.image_width, 2)
        # print('grid', grid.size()) #([64,28,28,2])
        transformed_x = grid_sample(x, grid)
        # print('transformed_x', transformed_x.size()) #([64,1,28,28])
        # print('logit', logit.size()) #([64,64])
        return transformed_x


def grid_sample(input, grid, canvas=None):
    output = F.grid_sample(input, grid)
    if canvas is None:
        return output
    else:
        input_mask = Variable(input.data.new(input.size()).fill_(1))
        output_mask = F.grid_sample(input_mask, grid)
        padded_output = output * output_mask + canvas * (1 - output_mask)
        return padded_output


# phi(x1, x2) = r^2 * log(r), where r = ||x1 - x2||_2
def compute_partial_repr(input_points, control_points):
    N = input_points.size(0)
    M = control_points.size(0)
    pairwise_diff = input_points.view(N, 1, 2) - control_points.view(1, M, 2)
    # original implementation, very slow
    # pairwise_dist = torch.sum(pairwise_diff ** 2, dim = 2) # square of distance
    pairwise_diff_square = pairwise_diff * pairwise_diff
    pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1]
    repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)
    # fix numerical error for 0 * log(0), substitute all nan with 0
    mask = repr_matrix != repr_matrix
    repr_matrix.masked_fill_(mask, 0)
    return repr_matrix


class TPSGridGen(nn.Module):
    def __init__(self, target_height, target_width, target_control_points):
        # print('target_control_points1', target_control_points.size()) ([16,2])
        super(TPSGridGen, self).__init__()
        assert target_control_points.ndimension() == 2
        assert target_control_points.size(1) == 2
        N = target_control_points.size(0)
        self.num_points = N
        target_control_points = target_control_points.float()

        # create padded kernel matrix
        forward_kernel = torch.zeros(N + 3, N + 3)
        target_control_partial_repr = compute_partial_repr(target_control_points, target_control_points)
        forward_kernel[:N, :N].copy_(target_control_partial_repr)
        forward_kernel[:N, -3].fill_(1)
        forward_kernel[-3, :N].fill_(1)
        forward_kernel[:N, -2:].copy_(target_control_points)
        forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))
        # compute inverse matrix
        inverse_kernel = torch.inverse(forward_kernel)

        # create target cordinate matrix
        HW = target_height * target_width
        target_coordinate = list(itertools.product(range(target_height), range(target_width)))
        target_coordinate = torch.Tensor(target_coordinate)  # HW x 2
        Y, X = target_coordinate.split(1, dim=1)
        Y = Y * 2 / (target_height - 1) - 1
        X = X * 2 / (target_width - 1) - 1
        target_coordinate = torch.cat([X, Y], dim=1)  # convert from (y, x) to (x, y)
        target_coordinate_partial_repr = compute_partial_repr(target_coordinate, target_control_points)
        target_coordinate_repr = torch.cat([
            target_coordinate_partial_repr, torch.ones(HW, 1), target_coordinate
        ], dim=1)

        # register precomputed matrices
        self.register_buffer('inverse_kernel', inverse_kernel)
        self.register_buffer('padding_matrix', torch.zeros(3, 2))
        self.register_buffer('target_coordinate_repr', target_coordinate_repr)

    def forward(self, source_control_points):
        assert source_control_points.ndimension() == 3
        # print('source_control_points.size(1)',source_control_points.size(1))
        # print('self.num_points', self.num_points)

        assert source_control_points.size(1) == self.num_points
        assert source_control_points.size(2) == 2
        batch_size = source_control_points.size(0)

        Y = torch.cat([source_control_points, Variable(self.padding_matrix.expand(batch_size, 3, 2))], 1)
        # print('Y',Y.size()) #([64,19,2])
        mapping_matrix = torch.matmul(Variable(self.inverse_kernel), Y)
        # print('mapping_matrix', mapping_matrix.size()) #([64,19,2])
        source_coordinate = torch.matmul(Variable(self.target_coordinate_repr), mapping_matrix)
        # print('target_coordinate_repr', self.target_coordinate_repr.size()) #([784,19])
        # print('source_coordinate', source_coordinate.size()) #([64,784,2])
        return source_coordinate


def get_model(opt):
    model = STNClsNet(opt)
    return model
