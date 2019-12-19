import numpy as np
import torch

from . import meter


class RetrievalMAPMeter(meter.Meter):
    MAP = 0
    PR = 1

    def __init__(self, topk=1000):
        self.topk = topk
        self.all_features = []
        self.all_lbs = []
        self.dis_mat = None
        self.mAP_value = None

    def reset(self):
        self.all_lbs.clear()
        self.all_features.clear()

    def add(self, features, lbs):
        self.all_features.append(features.cpu())
        self.all_lbs.append(lbs.cpu())

    def value(self, mode=MAP):
        if mode == self.MAP:
            return self.mAP()
        if mode == self.PR:
            return self.pr()
        raise NotImplementedError

    def mAP(self):
        if self.mAP_value is not None:
            return self.mAP_value
        if len(self.all_features) == 0:
            return 0
        fts = torch.cat(self.all_features).numpy()
        lbls = torch.cat(self.all_lbs).numpy()
        self.dis_mat = Eu_dis_mat_fast(np.mat(fts))
        num = len(lbls)
        # self.topk = num
        mAP = 0
        for i in range(num):
            scores = self.dis_mat[:, i]
            targets = (lbls == lbls[i]).astype(np.uint8)
            sortind = np.argsort(scores, 0)[:self.topk]
            truth = targets[sortind]
            sum = 0
            precision = []
            for j in range(self.topk):
                if truth[j]:
                    sum += 1
                    precision.append(sum * 1.0 / (j + 1))
            if len(precision) == 0:
                ap = 0
            else:
                # for ii in range(len(precision)):
                #     precision[ii] = max(precision[ii:])
                ap = np.array(precision).mean()
            mAP += ap
            # print(f'{i+1}/{num}\tap:{ap:.3f}\t')
        mAP = mAP / num
        self.mAP_value = mAP
        return self.mAP_value

    def pr(self):
        lbls = torch.cat(self.all_lbs).numpy()
        num = len(lbls)
        precisions = []
        recalls = []
        ans = []
        for i in range(num):
            scores = self.des_mat[:, i]
            targets = (lbls == lbls[i]).astype(np.uint8)
            sortind = np.argsort(scores, 0)[:self.topk]
            truth = targets[sortind]
            tmp = 0
            sum = truth[:self.topk].sum()
            precision = []
            recall = []
            for j in range(self.topk):
                if truth[j]:
                    tmp += 1
                    # precision.append(sum/(j + 1))
                recall.append(tmp * 1.0 / sum)
                precision.append(tmp * 1.0 / (j + 1))
            precisions.append(precision)
            for j in range(len(precision)):
                precision[j] = max(precision[j:])
            recalls.append(recall)
            tmp = []
            for ii in range(11):
                min_des = 100
                val = 0
                for j in range(self.topk):
                    if abs(recall[j] - ii * 0.1) < min_des:
                        min_des = abs(recall[j] - ii * 0.1)
                        val = precision[j]
                tmp.append(val)
            print('%d/%d' % (i + 1, num))
            ans.append(tmp)
        return np.array(ans).mean(0)


class CrossRetrievalMAPMeter(meter.Meter):
    MAP = 0
    PR = 1

    def __init__(self, topk=1000):
        self.topk = topk
        self.all_features_1 = []
        self.all_lbs_1 = []
        self.all_features_2 = []
        self.all_lbs_2 = []
        self.dis_mat_1 = None
        self.dis_mat_2 = None
        pass

    def reset(self):
        self.all_lbs_1.clear()
        self.all_features_1.clear()
        self.all_lbs_2.clear()
        self.all_features_2.clear()

    def add(self, features1, lbs1, features2, lbs2):
        self.all_features_1.append(features1.cpu())
        self.all_lbs_1.append(lbs1.cpu())
        self.all_features_2.append(features2.cpu())
        self.all_lbs_2.append(lbs2.cpu())

    def value(self, mode=MAP):
        if mode == self.MAP:
            return self.mAP()
        if mode == self.PR:
            return self.pr()
        raise NotImplementedError

    def mAP(self):
        fts1 = torch.cat(self.all_features_1).numpy()
        lbls1 = torch.cat(self.all_lbs_1).numpy()
        fts2 = torch.cat(self.all_features_2).numpy()
        lbls2 = torch.cat(self.all_lbs_2).numpy()
        self.dis_mat_1 = Eu_dis_mat_fast2(np.mat(fts2), np.mat(fts1))
        self.dis_mat_2 = self.dis_mat_1.T
        num = len(lbls1)
        mAP = 0
        for i in range(num):
            scores = self.dis_mat_1[:, i]
            targets = (lbls2 == lbls1[i]).astype(np.uint8)
            sortind = np.argsort(scores, 0)[:self.topk]
            truth = targets[sortind]
            sum = 0
            precision = []
            for j in range(self.topk):
                if truth[j]:
                    sum += 1
                    precision.append(sum * 1.0 / (j + 1))
            if len(precision) == 0:
                ap = 0
            else:
                ap = np.array(precision).mean()
            mAP += ap
            # print(f'{i+1}/{num}\tap:{ap:.3f}\t')
        mAP1 = mAP / num
        return mAP1

    def pr(self):
        lbls1 = torch.cat(self.all_lbs_1).numpy()
        lbls2 = torch.cat(self.all_lbs_2).numpy()
        num = len(lbls1)
        precisions = []
        recalls = []
        ans = []
        for i in range(num):
            scores = self.dis_mat_1[:, i]
            targets = (lbls2 == lbls1[i]).astype(np.uint8)
            sortind = np.argsort(scores, 0)[:self.topk]
            truth = targets[sortind]
            tmp = 0
            sum = truth[:self.topk].sum()
            precision = []
            recall = []
            for j in range(self.topk):
                if truth[j]:
                    tmp += 1
                    # precision.append(sum/(j + 1))
                recall.append(tmp * 1.0 / sum)
                precision.append(tmp * 1.0 / (j + 1))
            precisions.append(precision)
            for j in range(len(precision)):
                precision[j] = max(precision[j:])
            recalls.append(recall)
            tmp = []
            for ii in range(11):
                min_des = 100
                val = 0
                for j in range(self.topk):
                    if abs(recall[j] - ii * 0.1) < min_des:
                        min_des = abs(recall[j] - ii * 0.1)
                        val = precision[j]
                tmp.append(val)
            print('%d/%d' % (i + 1, num))
            ans.append(tmp)
        pr1 = np.array(ans).mean(0)
        num = len(lbls2)
        precisions = []
        recalls = []
        ans = []
        for i in range(num):
            scores = self.dis_mat_1[:, i]
            targets = (lbls1 == lbls2[i]).astype(np.uint8)
            sortind = np.argsort(scores, 0)[:self.topk]
            truth = targets[sortind]
            tmp = 0
            sum = truth[:self.topk].sum()
            precision = []
            recall = []
            for j in range(self.topk):
                if truth[j]:
                    tmp += 1
                    # precision.append(sum/(j + 1))
                recall.append(tmp * 1.0 / sum)
                precision.append(tmp * 1.0 / (j + 1))
            precisions.append(precision)
            for j in range(len(precision)):
                precision[j] = max(precision[j:])
            recalls.append(recall)
            tmp = []
            for ii in range(11):
                min_des = 100
                val = 0
                for j in range(self.topk):
                    if abs(recall[j] - ii * 0.1) < min_des:
                        min_des = abs(recall[j] - ii * 0.1)
                        val = precision[j]
                tmp.append(val)
            print('%d/%d' % (i + 1, num))
            ans.append(tmp)
        pr2 = np.array(ans).mean(0)
        return pr1, pr2


def Eu_dis_mat_fast(X):
    aa = np.sum(np.multiply(X, X), 1)
    ab = X * X.T
    D = aa + aa.T - 2 * ab
    D[D < 0] = 0
    # D = np.sqrt(D)
    D = np.maximum(D, D.T)
    return D
    # return -X * X.T


def Eu_dis_mat_fast2(X, Y):
    aa = np.sum(np.multiply(X, X), 1)
    bb = np.sum(np.multiply(Y, Y), 1)
    ab = X * Y.T
    D = aa + bb.T - 2 * ab
    D[D < 0] = 0
    # D = np.sqrt(D)
    return D
