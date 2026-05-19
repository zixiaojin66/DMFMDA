from torch import optim,nn
from tqdm import trange
from utils import k_matrix,make_adj
import dgl
import networkx as nx
import copy
import numpy as np
import torch as th
from sklearn.metrics import roc_auc_score,precision_recall_curve,auc,accuracy_score, precision_score, recall_score, f1_score,roc_curve
from sklearn.model_selection import KFold
import torch.nn.functional as F
import scipy.sparse as sp
from Model import Model
from CMD_Regularizer import cmd_regularizer
from DIF_Regularizer import dif_regularizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import random

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
#device = th.device("cpu")
kfolds=5

def print_met(list):
    print('AUC ：%.4f ' % (list[0]),
          'AUPR ：%.4f ' % (list[1]),
          'Accuracy ：%.4f ' % (list[2]),
          'precision ：%.4f ' % (list[3]),
          'recall ：%.4f ' % (list[4]),
          'f1_score ：%.4f \n' % (list[5]))

# 定义 Focal Loss 类
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_loss = nn.BCELoss(reduction='none')

    def forward(self, logits, labels):
        # 计算普通的二分类交叉熵损失
        bce_loss = self.bce_loss(logits, labels)

        # 计算预测概率
        probs = logits
        probs = probs * labels + (1 - probs) * (1 - labels)  # probs 是模型对正确类的预测概率

        # 计算 Focal Loss
        focal_loss = self.alpha * (1 - probs) ** self.gamma * bce_loss
        return focal_loss.mean()

def train(data,args):
    # ---------------------------------------------------#
    #   设置种子
    # ---------------------------------------------------#
    seed = 20827
    def seed_everything(seed=seed):
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False
    seed_everything()

    all_score = []
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=seed)
    train_idx, valid_idx = [], []
    for train_index, valid_index in kf.split(data['train_samples']):
        train_idx.append(train_index)
        valid_idx.append(valid_index)
    for i in range(kfolds):
        one_score = []
        model = Model().to(device)
        optimizer = optim.AdamW(model.parameters(), weight_decay=args.wd, lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
        #pos_weight = th.Tensor([1]).to(device)
        cross_entropy = FocalLoss()

        miRNA = data['ms']
        disease = data['ds']
        train_sample, valid_sample = data['train_samples'][train_idx[i]], data['train_samples'][valid_idx[i]]
        print(f'################Fold {i + 1} of {kfolds}################')
        epochs = trange(args.epochs, desc='train')
        for _ in epochs:
            model.train()
            optimizer.zero_grad()
            mm_matrix = k_matrix(data['ms'], args.neighbor)
            dd_matrix = k_matrix(data['ds'], args.neighbor)
            mm_nx = nx.from_numpy_array(mm_matrix)
            dd_nx = nx.from_numpy_array(dd_matrix)
            m_graph = dgl.from_networkx(mm_nx)
            d_graph = dgl.from_networkx(dd_nx)
            md_copy = copy.deepcopy(data['train_md'])
            md_copy[:, 1] = md_copy[:, 1] + args.miRNA_number
            md_graph = dgl.graph(
                (np.concatenate((md_copy[:, 0], md_copy[:, 1])), np.concatenate((md_copy[:, 1], md_copy[:, 0]))),
                num_nodes=args.miRNA_number + args.disease_number)
            miRNA_th = th.Tensor(miRNA)
            disease_th = th.Tensor(disease)
            train_score, m_c1, m_c2, m_s1, m_s2, d_c1, d_c2, d_s1, d_s2 = model(miRNA_th.to(device),
                                                                                disease_th.to(device),
                                                                                m_graph.to(device),
                                                                                d_graph.to(device),
                                                                                md_graph.to(device),
                                                                                train_sample)
            # train_score = model(miRNA_th.to(device),
            #                                                                     disease_th.to(device),
            #                                                                     m_graph.to(device),
            #                                                                     d_graph.to(device),
            #                                                                     md_graph.to(device),
            #                                                                     train_sample)
            train_samples = th.Tensor(train_sample).float()
            train_cross_loss = cross_entropy(th.flatten(train_score), train_samples[:, 2].to(device))
            train_cmd_loss = (cmd_regularizer(m_c1,m_c2,5) + cmd_regularizer(d_c1,d_c2,5))/(495*383)
            train_dif_loss = (dif_regularizer(m_c1,m_c2,m_s1,m_s2) + dif_regularizer(d_c1,d_c2,d_s1,d_s2))/(495*383)
            train_loss = train_cross_loss + 0.01*train_dif_loss + 0.01*train_cmd_loss
            print(train_cross_loss.item(),train_cmd_loss.item(),train_dif_loss.item(),train_loss.item())

            train_loss.backward()
            optimizer.step()
            # 在每个 epoch 结束时，更新学习率
            scheduler.step()


        model.eval()
        valid_score, m_c1, m_c2, m_s1, m_s2, d_c1, d_c2, d_s1, d_s2 = model(miRNA_th.to(device),
                                                                                disease_th.to(device),
                                                                                m_graph.to(device),
                                                                                d_graph.to(device),
                                                                                md_graph.to(device),
                                                                            valid_sample)
        # valid_score = model(miRNA_th.to(device),
        #                                                                         disease_th.to(device),
        #                                                                         m_graph.to(device),
        #                                                                         d_graph.to(device),
        #                                                                         md_graph.to(device),
        #                                                                     valid_sample)
        valid_score = valid_score.cpu().detach().numpy()
        scoree = valid_score
        sc_true = valid_sample[:, 2]
        fpr, tpr, thresholds = roc_curve(sc_true, scoree)
        # 选择最佳阈值
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        print("Best threshold：{:.4f}".format(optimal_threshold))

        # 计算auc
        aucc = roc_auc_score(sc_true, scoree)
        precision, recall, thresholds = precision_recall_curve(sc_true, scoree)
        print("AUC: {:.6f}".format(aucc))
        # plt.plot(recall, precision)
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.title('Precision-Recall Curve')
        # plt.show()

        auprc = auc(recall, precision)
        print("AUPRC: {:.6f}".format(auprc))

        scoree = np.array(scoree)
        # scoree=np.around(scoree, 0).astype(int)
        scoree = scoree.ravel()

        for i in range(len(scoree)):
            if scoree[i] >= optimal_threshold:
                scoree[i] = 1
            else:
                scoree[i] = 0
        accuracy = accuracy_score(sc_true, scoree)
        print("Accuracy: {:.6f}".format(accuracy))
        precision = precision_score(sc_true, scoree)
        print("Precision: {:.6f}".format(precision))
        recall = recall_score(sc_true, scoree)
        print("Recall: {:.6f}".format(recall))
        f1 = f1_score(sc_true, scoree)
        print("F1-score: {:.6f}".format(f1))
        # print(np.concatenate((data['m_num'][data['unsamples']],score),axis=1))
        one_score = [aucc, auprc, accuracy, precision, recall, f1]
        all_score.append(one_score)
    cv_metric = np.mean(all_score, axis=0)
    print('################5-Fold Result################')
    print_met(cv_metric)
