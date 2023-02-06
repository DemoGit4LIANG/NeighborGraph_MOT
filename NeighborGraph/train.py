import os

import torch
from torch.utils import data as Data
from torch.nn import functional as F
from tqdm import tqdm

from dataset import NeighborGraphDataSet, collate_fn
from model import GCN_Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cosine_loss(query, gallery, label, margin):
    target = (label.clone().detach() * 2 - 1).float().cuda()
    N = query.shape[0]
    loss = F.cosine_embedding_loss(query, gallery, target, margin, reduction='sum') / N

    return loss


def train_gcn(model, train_loader, loss_fn, optimizer, num_epoch=30):
    model.to(device)
    for epoch in range(num_epoch):
        train_loss = 0.
        N = 0
        for query, gallery, label in tqdm(train_loader):
            # aggregated by the gcn model
            query, gallery, sim = model(query.to(device), gallery.to(device))
            loss = loss_fn(query, gallery, label, margin=0.4)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            N += 1

        torch.save(model.state_dict(), os.path.join('/home/allen/ng_weights', 'model_{}.pt'.format(epoch + 1)))
        print('Epoch: {}, train_loss: {}'.format(epoch + 1, train_loss))

        if epoch == 10 or epoch == 20:
            for p in optimizer.param_groups:
                p['lr'] *= 0.5
                print('lr has decayed to:', p['lr'])


if __name__ == '__main__':
    data_root = '/home/allen/'
    train_set = NeighborGraphDataSet(data_root=data_root, num_neighbor=3)
    train_loader = Data.DataLoader(train_set, batch_size=8, collate_fn=collate_fn, shuffle=True, drop_last=True)
    gcn_model = GCN_Model(is_train=True, neighbor=3)
    optimizer = torch.optim.SGD(gcn_model.parameters(), lr=0.1)
    # optimizer = torch.optim.Adam(gcn_model.parameters(), lr=1e-3)

    train_gcn(model=gcn_model, train_loader=train_loader, loss_fn=cosine_loss, optimizer=optimizer, num_epoch=30)

