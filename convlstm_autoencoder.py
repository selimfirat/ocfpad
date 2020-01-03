import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from tqdm import tqdm

from convolution_lstm import ConvLSTMCell


class ConvLSTMAutoencoder():

    def __init__(self):
        self.model = ConvLSTMCell(input_channels=3, hidden_channels=3, kernel_size=3).cuda()
        self.optimizer = Adam(self.model.parameters(), lr=0.1) # default lr: 1e-3

        self.num_epochs = 50

    def train(self):

        data = [Variable(torch.randn(300, 3, 224, 224)).unsqueeze(1).cuda()]
        targets = [data[0].clone().cuda()]

        for i in range(self.num_epochs):
            for X, y_true in tqdm(zip(data, targets)):
                h, c = self.model.init_hidden(batch_size=1, hidden=3,
                                              shape=(224, 224))

                self.model.zero_grad()
                self.optimizer.zero_grad()

                y_pred = torch.empty(X.shape).cuda()

                for fidx in tqdm(range(X.shape[0])):
                    inp = X[fidx, :, :, :]

                    h, c = self.model.forward(inp, h, c)

                    y_pred[fidx, :, :, :] = h

                loss = torch.mean((y_pred - y_true) ** 2)

                print(loss)

                loss.backward(retain_graph=True)

                self.optimizer.step()


if __name__ == "__main__":
    stae = ConvLSTMAutoencoder()

    stae.train()
