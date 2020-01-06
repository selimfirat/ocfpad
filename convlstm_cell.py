import torch
from torch import nn
from torch import tanh, sigmoid, zeros
from torch.autograd import Variable


class ConvLSTMCell(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, cuda_idx):
        super().__init__()

        padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding, bias=True)
        self.Wxf = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding, bias=True)
        self.Wxc = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding, bias=True)
        self.Wxo = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding, bias=True)
        self.Whi = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.Whf = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.Whc = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.Who = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding, bias=False)

        self.Wi = nn.Parameter(Variable(zeros(1, out_channels, 224, 224)).cuda(cuda_idx))
        self.Wf = nn.Parameter(Variable(zeros(1, out_channels, 224, 224)).cuda(cuda_idx))
        self.Wo = nn.Parameter(Variable(zeros(1, out_channels, 224, 224)).cuda(cuda_idx))

    def forward(self, x, h, c):
        i = sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wi)
        f = sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wf)
        c_t = f * c + i * tanh(self.Wxc(x) + self.Whc(h))
        o = sigmoid(self.Wxo(x) + self.Who(h) + c_t * self.Wo)
        h_t = o * tanh(c_t)

        return h_t, c_t
