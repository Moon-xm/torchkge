# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>
"""

from torch import ones_like, zeros_like
from torch.nn import Module, Sigmoid
from torch.nn import MarginRankingLoss, SoftMarginLoss, BCELoss
import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginLoss(Module):
    """Margin loss as it was defined in `TransE paper
    <https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data>`_
    by Bordes et al. in 2013. This class implements :class:`torch.nn.Module`
    interface.

    """
    def __init__(self, margin, theta=0.001):
        super().__init__()
        # self.loss = MarginRankingLoss(margin=margin, reduction='sum')
        self.margin = nn.Parameter(torch.Tensor([margin]))
        self.margin.requires_grad = False
        self.theta = theta
        self.theta = nn.Parameter(torch.Tensor([theta]))
        self.theta.requires_grad = False

    def forward(self, positive_triplets, negative_triplets, point=None, n_point=None):
        """
        Parameters
        ----------
        positive_triplets: torch.Tensor, dtype: torch.float, shape: (b_size)
            Scores of the true triplets as returned by the `forward` methods of
            the models.
        negative_triplets: torch.Tensor, dtype: torch.float, shape: (b_size)
            Scores of the negative triplets as returned by the `forward`
            methods of the models.

        Returns
        -------
        loss: torch.Tensor, shape: (n_facts, dim), dtype: torch.float
            Loss of the form
            :math:`\\max\\{0, \\gamma - f(h,r,t) + f(h',r',t')\\}` where
            :math:`\\gamma` is the margin (defined at initialization),
            :math:`f(h,r,t)` is the score of a true fact and
            :math:`f(h',r',t')` is the score of the associated negative fact.
        """
        if (point is not None) and (n_point is not None):
            w_geo = self.calc_w(point, n_point)  # [batch_size]
            negative_triplets = w_geo*negative_triplets  # w_geo < 1 乘完后neg变大了 因为neg是负数

        return self.calc_loss(positive_triplets, negative_triplets)
        # return self.loss(positive_triplets, negative_triplets,
        #                  target=ones_like(positive_triplets))

    def calc_w(self, pos_points, neg_points):
        """
        :param pos_points: [batch_size, 2, 2]
        :param neg_points: [batch_size, 2, 2]
        :return: [batch_size]
        """
        dis_pos = self._calc_distance(pos_points)
        dis_neg = self._calc_distance(neg_points)  # [batch_size]
        log = torch.log10((dis_pos+self.theta)/(dis_neg+self.theta))
        w_geo = 1/(torch.abs(log)+1)   # 与正例相差越远 w越小
        # w_geo = torch.abs(log) + 1
        return w_geo

    def _calc_distance(self, points):
        """
        :param points: [batch_size, 2, 2]
        :return: [batch_size]
        """
        head, tail = points[:,0,:], points[:,1,:]  # [batch_size, 2]
        distance = torch.norm(head-tail, p=2, dim=1)  # 2 范数
        return distance

    def calc_loss(self,p_score, n_score):
        return torch.sum(F.relu(-p_score+n_score+self.margin))

class LogisticLoss(Module):
    """Logistic loss as it was defined in `TransE paper
    <https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data>`_
    by Bordes et al. in 2013. This class implements :class:`torch.nn.Module`
    interface.

    """
    def __init__(self):
        super().__init__()
        self.loss = SoftMarginLoss(reduction='sum')

    def forward(self, positive_triplets, negative_triplets):
        """
        Parameters
        ----------
        positive_triplets: torch.Tensor, dtype: torch.float, shape: (b_size)
            Scores of the true triplets as returned by the `forward` methods
            of the models.
        negative_triplets: torch.Tensor, dtype: torch.float, shape: (b_size)
            Scores of the negative triplets as returned by the `forward`
            methods of the models.
        Returns
        -------
        loss: torch.Tensor, shape: (n_facts, dim), dtype: torch.float
            Loss of the form :math:`\\log(1+ \\exp(\\eta \\times f(h,r,t))`
            where :math:`f(h,r,t)` is the score of the fact and :math:`\\eta`
            is either 1 or -1 if the fact is true or false.
        """
        targets = ones_like(positive_triplets)
        return self.loss(positive_triplets, targets) + \
            self.loss(negative_triplets, -targets)


class BinaryCrossEntropyLoss(Module):
    """This class implements :class:`torch.nn.Module` interface.

    """

    def __init__(self):
        super().__init__()
        self.sig = Sigmoid()
        self.loss = BCELoss(reduction='sum')

    def forward(self, positive_triplets, negative_triplets):
        """

        Parameters
        ----------
        positive_triplets: torch.Tensor, dtype: torch.float, shape: (b_size)
            Scores of the true triplets as returned by the `forward` methods
            of the models.
        negative_triplets: torch.Tensor, dtype: torch.float, shape: (b_size)
            Scores of the negative triplets as returned by the `forward`
            methods of the models.
        Returns
        -------
        loss: torch.Tensor, shape: (n_facts, dim), dtype: torch.float
            Loss of the form :math:`-\\eta \\cdot \\log(f(h,r,t)) +
            (1-\\eta) \\cdot \\log(1 - f(h,r,t))` where :math:`f(h,r,t)`
            is the score of the fact and :math:`\\eta` is either 1 or
            0 if the fact is true or false.
        """
        return self.loss(self.sig(positive_triplets),
                         ones_like(positive_triplets)) + \
            self.loss(self.sig(negative_triplets),
                      zeros_like(negative_triplets))


