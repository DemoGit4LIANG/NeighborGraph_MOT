import torch
import torch.nn as nn


class GCN(nn.Module):
    def __init__(self, is_train=True, neighbor=3):
        super(GCN, self).__init__()
        self.is_train = is_train
        self.neighbor = neighbor + 1

        a = torch.pow(torch.tensor(2. / self.neighbor), 0.5)
        self.A = torch.zeros(self.neighbor, self.neighbor)
        self.A[0][0] = 1.
        for i in range(self.neighbor - 1):
            self.A[0][i + 1] = a
            self.A[i + 1][0] = 1. / a
            self.A[i + 1][i + 1] = 1.
        self.A = self.A.unsqueeze(0).cuda()
        self.A.requires_grad = False

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
        self.fc = nn.Linear(in_features=2048, out_features=1024)
        # self.fc = nn.Linear(in_features=1024, out_features=256)

        if self.train:
            self.sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        else:
            self.sim = nn.CosineSimilarity(dim=0, eps=1e-6)

    def forward(self, query, gallery):
        gal = torch.bmm(self.A.expand(gallery.shape[0], self.neighbor, self.neighbor), gallery)
        gal = self.relu(self.W1(gal))
        gal = torch.bmm(self.A.expand(gal.shape[0], self.neighbor, self.neighbor), gal)
        gal = self.relu(self.W2(gal))
        gal = torch.bmm(self.A.expand(gal.shape[0], self.neighbor, self.neighbor), gal)
        gal = self.relu(self.W3(gal))

        gal = gal[:, 0, :].squeeze()
        gal = self.fc(gal)

        bs = query.shape[0]
        exp_size = int(gal.shape[0] / bs)
        query = torch.cat([item.expand(exp_size, 4, 256) for item in query], dim=0)

        que = torch.bmm(self.A.expand(query.shape[0], self.neighbor, self.neighbor), query)
        que = self.relu(self.W1(que))
        que = torch.bmm(self.A.expand(que.shape[0], self.neighbor, self.neighbor), que)
        que = self.relu(self.W2(que))
        que = torch.bmm(self.A.expand(que.shape[0], self.neighbor, self.neighbor), que)
        que = self.relu(self.W3(que))
        que = que[:, 0, :].squeeze()
        que = self.fc(que)

        return que, gal, self.sim(que, gal)


class GCN_Model(nn.Module):

    def __init__(self, is_train=True, neighbor=3):
        super(GCN_Model, self).__init__()
        self.is_train = is_train
        self.neighbor = neighbor

        self.GCN = GCN(neighbor=self.neighbor)

        if self.is_train:
            self.sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        else:
            self.sim = nn.CosineSimilarity(dim=0, eps=1e-6)

    def forward(self, query, gallery):
        que, gal, _ = self.GCN(query, gallery)
        return que, gal, self.sim(que, gal)
