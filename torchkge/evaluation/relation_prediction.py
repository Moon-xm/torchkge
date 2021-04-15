# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>
"""

from torch import empty
from tqdm.autonotebook import tqdm

from ..exceptions import NotYetEvaluatedError
from ..utils import DataLoader


class RelationPredictionEvaluator(object):
    """Evaluate performance of given embedding using link prediction method.

    Parameters
    ----------
    model: torchkge.models.interfaces.Model
        Embedding model inheriting from the right interface.
    knowledge_graph: torchkge.data_structures.KnowledgeGraph
        Knowledge graph on which the evaluation will be done.

    Attributes
    ----------
    model: torchkge.models.interfaces.Model
        Embedding model inheriting from the right interface.
    kg: torchkge.data_structures.KnowledgeGraph
        Knowledge graph on which the evaluation will be done.
    rank_true_heads: torch.Tensor, shape: (n_facts), dtype: `torch.int`
        For each fact, this is the rank of the true head when all entities
        are ranked as possible replacement of the head entity. They are
        ranked in decreasing order of scoring function :math:`f_r(h,t)`.
    rank_true_tails: torch.Tensor, shape: (n_facts), dtype: `torch.int`
        For each fact, this is the rank of the true tail when all entities
        are ranked as possible replacement of the tail entity. They are
        ranked in decreasing order of scoring function :math:`f_r(h,t)`.
    filt_rank_true_heads: torch.Tensor, shape: (n_facts), dtype: `torch.int`
        This is the same as the `rank_of_true_heads` when is the filtered
        case. See referenced paper by Bordes et al. for more information.
    filt_rank_true_tails: torch.Tensor, shape: (n_facts), dtype: `torch.int`
        This is the same as the `rank_of_true_tails` when is the filtered
        case. See referenced paper by Bordes et al. for more information.
    evaluated: bool
        Indicates if the method LinkPredictionEvaluator.evaluate has already
        been called.

    References
    ----------
    * Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston,
      and Oksana Yakhnenko.
      Translating Embeddings for Modeling Multi-relational Data.
      In Advances in Neural Information Processing Systems 26, pages 2787–2795,
      2013.
      https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data

    """

    def __init__(self, model, knowledge_graph):
        self.model = model
        self.kg = knowledge_graph

        # self.rank_true_heads = empty(size=(knowledge_graph.n_facts,)).long()
        # self.rank_true_tails = empty(size=(knowledge_graph.n_facts,)).long()
        # self.filt_rank_true_heads = empty(size=(knowledge_graph.n_facts,)
        #                                   ).long()
        # self.filt_rank_true_tails = empty(size=(knowledge_graph.n_facts,)
        #                                   ).long()
        self.rank_true_rel = empty(size=(knowledge_graph.n_facts,)).long()
        self.filt_rank_true_rel = empty(size=(knowledge_graph.n_facts,)
                                          ).long()

        self.evaluated = False

    def evaluate(self, b_size, verbose=False):
        """

        Parameters
        ----------
        b_size: int
            Size of the current batch.
        verbose: bool
            Indicates whether a progress bar should be displayed during
            evaluation.

        """
        use_cuda = next(self.model.parameters()).is_cuda

        if use_cuda:
            dataloader = DataLoader(self.kg, batch_size=b_size,
                                    use_cuda='batch')
            # self.rank_true_heads = self.rank_true_heads.cuda()
            # self.rank_true_tails = self.rank_true_tails.cuda()
            # self.filt_rank_true_heads = self.filt_rank_true_heads.cuda()
            # self.filt_rank_true_tails = self.filt_rank_true_tails.cuda()
            self.rank_true_rel = self.rank_true_rel.cuda()
            self.filt_rank_true_rel = self.filt_rank_true_rel.cuda()
        else:
            dataloader = DataLoader(self.kg, batch_size=b_size)

        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader),
                             unit='batch', disable=(not verbose),
                             desc='Relation prediction evaluation'):
            h_idx, t_idx, r_idx = batch[0], batch[1], batch[2]

            # rk_true_t, f_rk_true_t, rk_true_h, f_rk_true_h = \
            #     self.model.lp_helper(h_idx, t_idx, r_idx, self.kg)  # (filter)rank true ...
            rk_true_r, f_rk_true_r = self.model.rp_helper(h_idx, t_idx, r_idx, self.kg)  # (filter)rank true ...

            # self.rank_true_heads[i * b_size: (i + 1) * b_size] = rk_true_h
            # self.rank_true_tails[i * b_size: (i + 1) * b_size] = rk_true_t
            #
            # self.filt_rank_true_heads[i * b_size:
            #                           (i + 1) * b_size] = f_rk_true_h
            # self.filt_rank_true_tails[i * b_size:
            #                           (i + 1) * b_size] = f_rk_true_t
            self.rank_true_rel[i * b_size: (i + 1) * b_size] = rk_true_r
            self.filt_rank_true_rel[i * b_size: (i + 1) * b_size] = f_rk_true_r

        self.evaluated = True

        if use_cuda:
            # self.rank_true_heads = self.rank_true_heads.cpu()
            # self.rank_true_tails = self.rank_true_tails.cpu()
            # self.filt_rank_true_heads = self.filt_rank_true_heads.cpu()
            # self.filt_rank_true_tails = self.filt_rank_true_tails.cpu()
            self.rank_true_rel = self.rank_true_rel.cpu()
            self.filt_rank_true_rel = self.filt_rank_true_rel.cpu()

    def mean_rank(self):
        """

        Returns
        -------
        mean_rank: float
            Mean rank of the true entity when replacing alternatively head
            and tail in any fact of the dataset.
        filt_mean_rank: float
            Filtered mean rank of the true entity when replacing
            alternatively head and tail in any fact of the dataset.

        """
        if not self.evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated call '
                                       'LinkPredictionEvaluator.evaluate')
        # sum_ = (self.rank_true_heads.float().mean() +
        #         self.rank_true_tails.float().mean()).item()
        # filt_sum = (self.filt_rank_true_heads.float().mean() +
        #             self.filt_rank_true_tails.float().mean()).item()
        mr = self.rank_true_rel.float().mean().item()
        filt_mr = self.filt_rank_true_rel.float().mean().item()
        return mr, filt_mr

    # def hit_at_k_heads(self, k=10):
    #     if not self.evaluated:
    #         raise NotYetEvaluatedError('Evaluator not evaluated call '
    #                                    'LinkPredictionEvaluator.evaluate')
    #     head_hit = (self.rank_true_heads <= k).float().mean()
    #     filt_head_hit = (self.filt_rank_true_heads <= k).float().mean()
    #
    #     return head_hit.item(), filt_head_hit.item()
    #
    # def hit_at_k_tails(self, k=10):
    #     if not self.evaluated:
    #         raise NotYetEvaluatedError('Evaluator not evaluated call '
    #                                    'LinkPredictionEvaluator.evaluate')
    #     tail_hit = (self.rank_true_tails <= k).float().mean()
    #     filt_tail_hit = (self.filt_rank_true_tails <= k).float().mean()
    #
    #     return tail_hit.item(), filt_tail_hit.item()

    def hit_at_k_rel(self, k=1):
        if not self.evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated call '
                                       'LinkPredictionEvaluator.evaluate')
        rel_hit = (self.rank_true_rel <= k).float().mean()
        filt_rel_hit = (self.filt_rank_true_rel <= k).float().mean()

        return rel_hit.item(), filt_rel_hit.item()


    def hit_at_k(self, k=10):
        """

        Parameters
        ----------
        k: int
            Hit@k is the number of entities that show up in the top k that
            give facts present in the dataset.

        Returns
        -------
        avg_hitatk: float
            Average of hit@k for head and tail replacement.
        filt_avg_hitatk: float
            Filtered average of hit@k for head and tail replacement.

        """
        if not self.evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated call '
                                       'LinkPredictionEvaluator.evaluate')

        # head_hit, filt_head_hit = self.hit_at_k_heads(k=k)
        # tail_hit, filt_tail_hit = self.hit_at_k_tails(k=k)
        rel_hit, filt_rel_hit = self.hit_at_k_rel(k=k)

        # return (head_hit + tail_hit) / 2, (filt_head_hit + filt_tail_hit) / 2
        return rel_hit, filt_rel_hit

    def mrr(self):
        """

        Returns
        -------
        avg_mrr: float
            Average of mean recovery rank for head and tail replacement.
        filt_avg_mrr: float
            Filtered average of mean recovery rank for head and tail
            replacement.

        """
        if not self.evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated call '
                                       'LinkPredictionEvaluator.evaluate')
        # head_mrr = (self.rank_true_heads.float()**(-1)).mean()
        # tail_mrr = (self.rank_true_tails.float()**(-1)).mean()
        # filt_head_mrr = (self.filt_rank_true_heads.float()**(-1)).mean()
        # filt_tail_mrr = (self.filt_rank_true_tails.float()**(-1)).mean()

        rel_mrr = (self.rank_true_rel.float() ** (-1)).mean()
        filt_rel_mrr = (self.filt_rank_true_rel.float()**(-1)).mean()
        # return ((head_mrr + tail_mrr).item() / 2,
        #         (filt_head_mrr + filt_tail_mrr).item() / 2)
        return (rel_mrr, filt_rel_mrr)

    def print_results(self, k=None):
        """

        Parameters
        ----------
        k: int or list
            k (or list of k) such that hit@k will be printed.
        """
        if k is None:
            k = 1

        if k is not None and type(k) == int:
            print('Relation prediction:')
            print('Hit@{} : {:>5.2%} \t\t Filt. Hit@{} : {:>5.2%}'.format(
                k, self.hit_at_k(k=k)[0],
                k, self.hit_at_k(k=k)[1]))
        if k is not None and type(k) == list:
            for i in k:
                print('Hit@{} : {} \t\t Filt. Hit@{} : {}'.format(
                    i, self.hit_at_k(k=i)[0],
                    i, self.hit_at_k(k=i)[1]))
                    # i, round(self.hit_at_k(k=i)[0], n_digits),
                    # i, round(self.hit_at_k(k=i)[1], n_digits)))

        print('Mean Rank : {:.2f} \t Filt. Mean Rank : {:.2f}'.format(
            self.mean_rank()[0], self.mean_rank()[1]))
        # print('MRR : {} \t\t Filt. MRR : {}'.format(
        #     round(self.mrr()[0], n_digits), round(self.mrr()[1], n_digits)))
