import os
import random

import numpy as np
import torch
from torch.utils import data as Data

from utils import readpkl, bbox_ctr_dist


class NeighborGraphDataSet(Data.Dataset):

    def find_nearest_neighbor(self, target, cands, num_neighbor=3):
        # target: [bbox, feat]
        # cands: [[bbox, feat],..., [bbox, feat]]
        feat_n = [target[1]]
        if len(cands) == 0:
            return target[1], np.array(feat_n * (num_neighbor + 1))

        index = np.zeros(len(cands))
        for i, c in enumerate(cands):
            index[i] = bbox_ctr_dist(target[0], c[0])
        index = [[dist, i] for i, dist in enumerate(index)]
        index = sorted(index, key=lambda i: i[0])

        for i in range(num_neighbor):
            if i > (len(index) - 1):
                feat_n.append(target[1])
            else:
                feat_n.append(cands[index[i][1]][1])

        return target[1], np.asarray(feat_n)

    def __init__(self, data_root, num_neighbor=3):
        self.num_neighbor = num_neighbor
        # type: dict
        # {id: {fno: [bbox, feat]}}
        print('### load train data from {}'.format(os.path.join(data_root, 'train_data.pkl')))
        self.train_data = readpkl(os.path.join(data_root, 'train_data.pkl'))
        # print(os.path.join(data_root, 'train_data.pkl'))
        # {frame_no: id_list}
        self.fno_ids = self.gen_fno_ids(self.train_data)
        self.id_list = list(self.train_data.keys())
        # for bug fix
        self.all_fno_list = list(self.fno_ids.keys())

        self.pos_num = 5
        self.neg_num = 95

    def __getitem__(self, id_index):
        id = self.id_list[id_index]
        # {fno: [bbox, feat]}
        target_tracks = self.train_data[id]
        fno_list = list(target_tracks.keys())
        if self.pos_num + 1 >= len(fno_list):
            selected_fno = random.sample(fno_list, len(fno_list))
            # bug fixed
            selected_fno += [selected_fno[-1]] * (self.pos_num + 1 - len(fno_list))
        else:
            # selected_fno = random.sample(fno_list, self.pos_num)
            selected_fno = random.sample(fno_list, self.pos_num + 1)
        pos, neg = [], []
        label = []
        for s_fno in selected_fno:
            target = target_tracks[s_fno]
            cand_ids = self.fno_ids[s_fno]
            # remove the target id
            cand_ids = [c_id for c_id in cand_ids if c_id != id]
            cands = [self.train_data[c_id][s_fno] for c_id in cand_ids]
            # target and neighbor features
            # shape: e.g., 4 x 256
            # feat_t: 256
            # feat_n: 4x256, feat_n[0] = feat_t
            feat_t, feat_n = self.find_nearest_neighbor(target, cands, num_neighbor=self.num_neighbor)
            pos_neighbor_graph = feat_n
            pos.append(pos_neighbor_graph)
            label.append(1.)

            for c_id in cand_ids:
                if len(neg) >= self.neg_num:
                    break
                target = self.train_data[c_id][s_fno]
                cand_ids = [cc_id for cc_id in cand_ids if cc_id != c_id]
                cands = [self.train_data[cc_id][s_fno] for cc_id in cand_ids]
                feat_t, feat_n = self.find_nearest_neighbor(target, cands, num_neighbor=self.num_neighbor)
                neg_neighbor_graph = feat_n
                neg.append(neg_neighbor_graph)
                label.append(0.)

        if len(pos) < self.pos_num:
            remain_pos = self.pos_num - len(pos)
            for _ in range(remain_pos):
                pos.append(pos[-1])
                label.append(1.)

        if len(neg) < self.neg_num:
            remain_neg = self.neg_num - len(neg)
            # remove the selected frame in the previous step
            selected_fno_set = set(selected_fno)
            for s_fno in selected_fno_set:
                fno_list.remove(s_fno)

            # for bug fix
            if remain_neg > len(fno_list):
                fno_list += random.sample(self.all_fno_list, 100)

            selected_fno = random.sample(fno_list, remain_neg)

            for s_fno in selected_fno:
                if len(neg) >= self.neg_num:
                    break
                cand_ids = self.fno_ids[s_fno]
                # remove the target id
                cand_ids = [c_id for c_id in cand_ids if c_id != id]
                cands = [self.train_data[c_id][s_fno] for c_id in cand_ids]
                feat_t, feat_n = self.find_nearest_neighbor(target, cands, num_neighbor=self.num_neighbor)
                neg_neighbor_graph = feat_n
                neg.append(neg_neighbor_graph)
                label.append(0.)

        query = torch.tensor(pos.pop(0))
        _ = label.pop(0)

        pos = torch.tensor(pos)
        neg = torch.tensor(neg)

        gallery = torch.cat((pos, neg), dim=0)

        return query, gallery, torch.tensor(label)

    def __len__(self):
        return len(self.id_list)

    def gen_fno_ids(self, data):
        fno_ids = dict()
        for id, tracks in data.items():
            for fno, track in tracks.items():
                if fno in fno_ids:
                    fno_ids[fno].append(id)
                else:
                    fno_ids[fno] = [id]

        return fno_ids


def collate_fn(batch):
    poss, samples, labels = zip(*batch)
    poss = torch.stack(poss)
    samples = torch.cat(samples, 0)
    labels = torch.cat(labels, 0)

    return poss, samples, labels
