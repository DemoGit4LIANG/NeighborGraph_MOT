import torch
import torch.nn as nn


class GCN(nn.Module):
    def __init__(self, is_train=True, neighbor=4):
        super(GCN, self).__init__()
        self.is_train = is_train
        self.neighbor = neighbor

        a = torch.pow(torch.tensor(2. / self.neighbor), 0.5)
        self.A = torch.zeros(self.neighbor, self.neighbor)
        self.A[0][0] = 1.
        for i in range(self.neighbor - 1):
            self.A[0][i + 1] = a
            self.A[i + 1][0] = 1. / a
            self.A[i + 1][i + 1] = 1.
        self.A = self.A.unsqueeze(0).cuda()
        self.A.requires_grad = False

        '''
        self.W1 = nn.Linear(in_features=256, out_features=512)
        self.W1.bias.data.fill_(0.)
        self.W1.bias.requires_grad = False
        self.W2 = nn.Linear(in_features=512, out_features=1024)
        self.W2.bias.requires_grad = False
        self.W2.bias.data.fill_(0.)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_features=1024, out_features=1024)
        '''
        self.W1 = nn.Linear(in_features=256, out_features=512)
        self.W1.bias.data.fill_(0.)
        self.W1.bias.requires_grad = False
        self.W2 = nn.Linear(in_features=512, out_features=1024)
        self.W2.bias.requires_grad = False
        self.W2.bias.data.fill_(0.)
        self.W3 = nn.Linear(in_features=1024, out_features=2048)
        self.W3.bias.requires_grad = False
        self.W3.bias.data.fill_(0.)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_features=2048, out_features=2048)
        # self.fc = nn.Linear(in_features=1024, out_features=256)

    def forward(self, input):

        output = torch.bmm(self.A.expand(input.shape[0], self.neighbor, self.neighbor), input)
        output = self.relu(self.W1(output))
        output = torch.bmm(self.A.expand(output.shape[0], self.neighbor, self.neighbor), output)
        output = self.relu(self.W2(output))
        output = torch.bmm(self.A.expand(output.shape[0], self.neighbor, self.neighbor), output)
        output = self.relu(self.W3(output))
        output = output[:, 0, :].squeeze()
        output = self.fc(output)

        return output


class GCN_Model(nn.Module):
    def __init__(self, is_train=False, neighbor=4):
        super(GCN_Model, self).__init__()
        self.is_train = is_train
        self.neighbor = neighbor

        self.GCN = GCN(neighbor=self.neighbor)

    def forward(self, input):
        input = self.GCN(input)
        return input


if __name__ == '__main__':
    ng = GCN_Model(is_train=False, neighbor=4).cuda()
    input = torch.ones(1, 4, 256).cuda()
    r = ng(input)
    print(r.shape)
