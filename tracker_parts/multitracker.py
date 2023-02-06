import numpy as np
from numba import jit
from collections import deque
import itertools
import os
import os.path as osp
import time
import torch
import cv2
import torch.nn.functional as F
from scipy.spatial.distance import cdist

from src.lib.models.model import create_model, load_model
from src.lib.models.decode import mot_decode
from src.lib.tracking_utils.utils import *
from src.lib.tracking_utils.log import logger
from src.lib.tracking_utils.kalman_filter import KalmanFilter
from src.lib.models import *
from src.lib.tracker import matching
from .basetrack import BaseTrack, TrackState
from src.lib.utils.post_process import ctdet_post_process
from src.lib.utils.image import get_affine_transform
from src.lib.models.utils import _tranpose_and_gather_feat
from ..models.gcn.neighbor_model import GCN_Model


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, temp_feat, buffer_size=30):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9

        self.tmp_cnt = 0

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.frame_list = [frame_id]

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.frame_list.append(frame_id)
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.frame_list.append(frame_id)
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class JDETracker(object):
    def __init__(self, opt, frame_rate=30, gcn_model=None):
        self.opt = opt
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')
        print('Creating model...')
        self.model = create_model(opt.arch, opt.heads, opt.head_conv)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        # self.det_thresh = opt.conf_thres
        self.det_thresh = opt.conf_thres + 0.1
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size
        self.max_per_image = opt.K
        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)

        self.kalman_filter = KalmanFilter()

        self.gcn = gcn_model

    def post_process(self, dets, meta):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt.num_classes)
        for j in range(1, self.opt.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.opt.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)

        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.opt.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.opt.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]

        return results

    def update(self, im_blob, img0):

        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = im_blob.shape[2]
        inp_width = im_blob.shape[3]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}

        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            output = self.model(im_blob)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            id_feature = output['id']
            id_feature = F.normalize(id_feature, dim=1)

            reg = output['reg'] if self.opt.reg_offset else None
            dets, inds = mot_decode(hm, wh, reg=reg, ltrb=self.opt.ltrb, K=self.opt.K)

            id_feature = _tranpose_and_gather_feat(id_feature, inds)
            id_feature = id_feature.squeeze(0)
            id_feature = id_feature.cpu().numpy()

        dets = self.post_process(dets, meta)
        dets = self.merge_outputs([dets])[1]

        # conf_thres: 0.4
        remain_inds = dets[:, 4] > 0.4
        dets = dets[remain_inds]
        id_feature = id_feature[remain_inds]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
                          (tlbrs, f) in zip(dets[:, :5], id_feature)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with IOU'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        # dists = matching.iou_distance(strack_pool, detections)
        iou_dists = matching.iou_distance(strack_pool, detections)
        emb_dists = matching.embedding_distance(strack_pool, detections)

        dists = 0.5 * iou_dists + 0.5 * emb_dists
        dists[iou_dists > 0.8] = np.inf

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.opt.match_thres)

        m_track = []
        for itracked, idet in matches:
            if emb_dists[itracked, idet] > 0.4:
                u_track = np.append(u_track, itracked)
                u_detection = np.append(u_detection, idet)
                continue
            m_track.append(itracked)

            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        u_track = np.sort(u_track)
        u_detection = np.sort(u_detection)

        if len(u_detection) > 0:
            dets_second = dets[u_detection]
            id_feature_second = id_feature[u_detection]

        # association the untrack to the low score detections
        if len(u_detection) > 0 and len(u_track) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
                                 (tlbrs, f) in zip(dets_second[:, :5], id_feature_second)]
        else:
            detections_second = []

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]

        # try to the second association (by neighbor graph)
        track_graph_feats, track_det_feats = None, None
        if len(r_tracked_stracks) > 0 and len(detections_second) > 0:
            m_tracked_stracks = [strack_pool[i] for i in m_track if strack_pool[i].state == TrackState.Tracked]
            _, track_ng_feat = build_track_neighbor_graph(r_tracked_stracks,
                                                                 m_tracked_stracks,
                                                                 num_neighbor=3)
            track_graph_feats = self.gcn(torch.tensor(track_ng_feat).cuda()).detach().cpu().numpy()
            if len(track_graph_feats.shape) == 1:
                track_graph_feats = track_graph_feats[None, :]

            _, det_ng_feat = build_det_neighbor_graph(detections_second, detections,
                                                           num_neighbor=3)
            track_det_feats = self.gcn(torch.tensor(det_ng_feat).cuda()).detach().cpu().numpy()
            if len(track_det_feats.shape) == 1:
                track_det_feats = track_det_feats[None, :]

        # dists = matching.iou_distance(r_tracked_stracks, detections_second)
        # iou_dists = matching.iou_distance(r_tracked_stracks, detections_second)

        if track_graph_feats is not None and track_det_feats is not None:
            emb_dists = embedding_cosine_dist(track_graph_feats, track_det_feats)
        else:
            emb_dists = matching.embedding_distance(r_tracked_stracks, detections_second)
        dists = emb_dists
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.2)
        # Second association (by neighbor graph) ends

        # record the matched track id
        matched_trk_by_ng = set()
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            # record the track_id
            matched_trk_by_ng.add(track.track_id)
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            # det_thresh: 0.5
            if track.score < 0.5:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # self.tracked_stracks = remove_fp_stracks(self.tracked_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))

        return output_stracks, matched_trk_by_ng


def eculid_dist(ctr_a, ctr_b):
    return ((ctr_a[0] - ctr_b[0]) ** 2 + (ctr_a[1] - ctr_b[1]) ** 2) ** 0.5


def embedding_cosine_dist(track_feats, det_feats):
    cost_matrix = np.zeros((len(track_feats), len(det_feats)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    cost_matrix = np.maximum(0.0, cdist(track_feats, det_feats, 'cosine'))  # Nomalized features
    return cost_matrix


def build_det_neighbor_graph(u_dets, dets, num_neighbor=3):
    all_neighbor_graph = []
    all_neighbor_graph_feat = []
    for u in u_dets:
        neighbor_graph = []
        neighbor_graph_feat = []
        u_ctr = u.to_xyah()[:2]
        ctr_list = [d.to_xyah()[:2] for d in dets]
        ctr_dists = [[eculid_dist(u_ctr, d_ctr), i] for i, d_ctr in enumerate(ctr_list)]
        ctr_dists = sorted(ctr_dists, key=lambda i: i[0])
        ctr_dists = ctr_dists[:num_neighbor + 1]
        r = num_neighbor + 1 - len(ctr_dists)

        if r > 0:
            for i in range(r):
                ctr_dists.insert(0, ctr_dists[0])

        for dist, i in ctr_dists:
            neighbor_graph.append(dets[i])
            neighbor_graph_feat.append(dets[i].curr_feat)

        all_neighbor_graph.append(neighbor_graph)
        all_neighbor_graph_feat.append(neighbor_graph_feat)

    return all_neighbor_graph, np.stack(all_neighbor_graph_feat, axis=0)


def build_track_neighbor_graph(u_stracks, m_stracks, num_neighbor=3):
    all_neighbor_graph = []
    all_neighbor_graph_feat = []
    for u in u_stracks:
        neighbor_graph = [u]
        neighbor_graph_feat = [u.smooth_feat]
        last_frame = u.frame_id
        temp_cands = []
        for m in m_stracks:
            if last_frame in m.frame_list:
                temp_cands.append(m)

        u_ctr = u.to_xyah()[:2]
        ctr_list = [c.to_xyah()[:2] for c in temp_cands]
        ctr_dists = [[eculid_dist(u_ctr, m_ctr), i] for i, m_ctr in enumerate(ctr_list)]
        ctr_dists = sorted(ctr_dists, key=lambda i: i[0])
        ctr_dists = ctr_dists[:num_neighbor]

        r = num_neighbor - len(ctr_dists)
        if r > 0:
            for i in range(r):
                ctr_dists.insert(0, ctr_dists[0])

        for dist, i in ctr_dists:
            neighbor_graph.append(temp_cands[i])
            neighbor_graph_feat.append(temp_cands[i].smooth_feat)

        all_neighbor_graph.append(neighbor_graph)
        all_neighbor_graph_feat.append(neighbor_graph_feat)

    return all_neighbor_graph, np.stack(all_neighbor_graph_feat, axis=0)


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb


def remove_fp_stracks(stracksa, n_frame=10):
    remain = []
    for t in stracksa:
        score_5 = t.score_list[-n_frame:]
        score_5 = np.array(score_5, dtype=np.float32)
        index = score_5 < 0.45
        num = np.sum(index)
        if num < n_frame:
            remain.append(t)
    return remain
