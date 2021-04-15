# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>
"""

from collections import defaultdict
from torch import cat, eq, int64, long, randperm, tensor, Tensor, zeros_like
from torch.utils.data import Dataset

from torchkge.exceptions import SizeMismatchError, WrongArgumentsError, SanityError
from torchkge.utils.operations import get_dictionaries
from torchkge.utils.my_utils import safe_strip
import pandas as pd
import numpy as np

class KnowledgeGraph(Dataset):
    """Knowledge graph representation. At least one of `df` and `kg`
    parameters should be passed.

    Parameters
    ----------
    df: pandas.DataFrame (optional)
        Data frame containing three columns [from, to, rel].
    kg: dict, optional
        Dictionary with keys ('heads', 'tails', 'relations') and values
        the corresponding torch long tensors.
    ent2ix: dict, optional
        Dictionary mapping entity labels to their integer key. This is
        computed if not passed as argument.
    rel2ix: dict, optional
        Dictionary mapping relation labels to their integer key. This is
        computed if not passed as argument.
    dict_of_heads: dict, optional
        Dictionary of possible heads :math:`h` so that the triple
        :math:`(h,r,t)` gives a true fact. The keys are tuples (t, r).
        This is computed if not passed as argument.
    dict_of_tails: dict, optional
        Dictionary of possible tails :math:`t` so that the triple
        :math:`(h,r,t)` gives a true fact. The keys are tuples (h, r).
        This is computed if not passed as argument.


    Attributes
    ----------
    ent2ix: dict
        Dictionary mapping entity labels to their integer key.
    rel2ix: dict
        Dictionary mapping relation labels to their integer key.
    n_ent: int
        Number of distinct entities in the data set.
    n_rel: int
        Number of distinct entities in the data set.
    n_facts: int
        Number of samples in the data set. A sample is a fact: a triplet
        (h, r, l).
    head_idx: torch.Tensor, dtype = torch.long, shape: (n_facts)
        List of the int key of heads for each fact.
    tail_idx: torch.Tensor, dtype = torch.long, shape: (n_facts)
        List of the int key of tails for each fact.
    relations: torch.Tensor, dtype = torch.long, shape: (n_facts)
        List of the int key of relations for each fact.

    """

    def __init__(self, df=None, kg=None, ent2ix=None, rel2ix=None,
                 dict_of_heads=None, dict_of_tails=None, dict_of_rel=None, id2point=None, geo=None):

        if df is None:
            if kg is None:
                raise WrongArgumentsError("Please provide at least one "
                                          "argument of `df` and kg`")
            else:
                try:
                    assert (type(kg) == dict) & ('heads' in kg.keys()) & \
                           ('tails' in kg.keys()) & \
                           ('relations' in kg.keys())
                except AssertionError:
                    raise WrongArgumentsError("Keys in the `kg` dict should "
                                              "contain `heads`, `tails`, "
                                              "`relations`.")
                try:
                    assert (rel2ix is not None) & (ent2ix is not None)
                except AssertionError:
                    raise WrongArgumentsError("Please provide the two "
                                              "dictionaries ent2ix and rel2ix "
                                              "if building from `kg`.")
        else:
            if kg is not None:
                raise WrongArgumentsError("`df` and kg` arguments should not "
                                          "both be provided.")

        if ent2ix is None:
            self.ent2ix = get_dictionaries(df, ent=True)
        else:
            self.ent2ix = ent2ix

        if rel2ix is None:
            self.rel2ix = get_dictionaries(df, ent=False)
        else:
            self.rel2ix = rel2ix

        if id2point is not None:
            self.id2point = id2point

        self.n_ent = max(self.ent2ix.values()) + 1
        self.n_rel = max(self.rel2ix.values()) + 1
        self.geo = geo

        if df is not None:
            # build kg from a pandas dataframe
            self.n_facts = len(df)
            self.head_idx = tensor(df['from'].map(self.ent2ix).values).long()
            self.tail_idx = tensor(df['to'].map(self.ent2ix).values).long()
            self.relations = tensor(df['rel'].map(self.rel2ix).values).long()
        else:
            # build kg from another kg
            self.n_facts = kg['heads'].shape[0]
            self.head_idx = kg['heads']
            self.tail_idx = kg['tails']
            self.relations = kg['relations']
            try:
                self.point = kg['point']
            except:
                pass

        if (geo is not None) and (df is not None):  # Geo
            self.entity2point, self.id2point = self.load_point(geo)
            self.point = np.array([[self.entity2point[triplet[0]], self.entity2point[triplet[2]]] for triplet in df.values])


        if dict_of_heads is None or dict_of_tails is None or dict_of_rel is None:
            self.dict_of_heads = defaultdict(set)
            self.dict_of_tails = defaultdict(set)
            self.dict_of_rel = defaultdict(set)
            self.evaluate_dicts()

        else:
            self.dict_of_heads = dict_of_heads
            self.dict_of_tails = dict_of_tails
            self.dict_of_rel = dict_of_rel
        try:
            self.sanity_check()
        except AssertionError:
            raise SanityError("Please check the sanity of arguments.")

    def __len__(self):
        return self.n_facts

    def __getitem__(self, item):
        if self.geo is not None:
            return (self.head_idx[item].item(),
                    self.tail_idx[item].item(),
                    self.relations[item].item(),
                    self.point[item])
        else:
            return (self.head_idx[item].item(),
                    self.tail_idx[item].item(),
                    self.relations[item].item())

    def load_point(self, geo):
        """
        generate point data
        :return: dict of point data  eg:{'Beijing': [116.51288500000001, 39.847469], 'Chongqing': [116.41338400000001, 39.910925],...}
        """
        point_data = pd.read_csv(geo, sep='\t', index_col=False, encoding='utf-8')
        point_data = point_data.applymap(lambda x: safe_strip(x))
        point_data['id'] = point_data['name'].map(lambda x: self.ent2ix[x])

        id_ls = list(point_data['id'])
        entity_ls = list(point_data['name'])
        long_ls = list(point_data['long'])
        lat_ls = list(point_data['lat'])
        long_lat_ls = [[x, y] for x, y in zip(long_ls, lat_ls)]
        ent2point_dic = zip(entity_ls, long_lat_ls)
        ent2point_dic = dict(ent2point_dic)
        id2point_dic = zip(id_ls, long_lat_ls)
        id2point_dic = dict(id2point_dic)
        return ent2point_dic, id2point_dic

    def sanity_check(self):
        assert (type(self.dict_of_heads) == defaultdict) & \
               (type(self.dict_of_tails) == defaultdict) &\
               (type(self.dict_of_rel) == defaultdict)
        assert (type(self.ent2ix) == dict) & (type(self.rel2ix) == dict)
        assert (len(self.ent2ix) == self.n_ent) & \
               (len(self.rel2ix) == self.n_rel)
        assert (type(self.head_idx) == Tensor) & \
               (type(self.tail_idx) == Tensor) & \
               (type(self.relations) == Tensor)
        assert (self.head_idx.dtype == int64) & \
               (self.tail_idx.dtype == int64) & (self.relations.dtype == int64)
        assert (len(self.head_idx) == len(self.tail_idx) == len(self.relations))

    def split_kg(self, share=0.8, sizes=None, validation=False, geo=None):
        """Split the knowledge graph into train and test. If `sizes` is
        provided then it is used to split the samples as explained below. If
        only `share` is provided, the split is done at random but it assures
        to keep at least one fact involving each type of entity and relation
        in the training subset.

        Parameters
        ----------
        share: float
            Percentage to allocate to train set.
        sizes: tuple
            Tuple of ints of length 2 or 3.

            * If len(sizes) == 2, then the first sizes[0] values of the
              knowledge graph will be used as training set and the rest as
              test set.

            * If len(sizes) == 3, then the first sizes[0] values of the
              knowledge graph will be used as training set, the following
              sizes[1] as validation set and the last sizes[2] as testing set.
        validation: bool
            Indicate if a validation set should be produced along with train
            and test sets.

        Returns
        -------
        train_kg: torchkge.data_structures.KnowledgeGraph
        val_kg: torchkge.data_structures.KnowledgeGraph, optional
        test_kg: torchkge.data_structures.KnowledgeGraph

        """
        if sizes is not None:
            try:
                if len(sizes) == 3:
                    try:
                        assert (sizes[0] + sizes[1] + sizes[2] == self.n_facts)
                    except AssertionError:
                        raise WrongArgumentsError('Sizes should sum to the '
                                                  'number of facts.')
                elif len(sizes) == 2:
                    try:
                        assert (sizes[0] + sizes[1] == self.n_facts)
                    except AssertionError:
                        raise WrongArgumentsError('Sizes should sum to the '
                                                  'number of facts.')
                else:
                    raise SizeMismatchError('Tuple `sizes` should be of '
                                            'length 2 or 3.')
            except AssertionError:
                raise SizeMismatchError('Tuple `sizes` should sum up to the '
                                        'number of facts in the knowledge '
                                        'graph.')
        else:
            assert share < 1

        if ((sizes is not None) and (len(sizes) == 3)) or \
                ((sizes is None) and validation):
            # return training, validation and a testing graphs

            if (sizes is None) and validation:
                mask_tr, mask_val, mask_te = self.get_mask(share,
                                                           validation=True)
            else:
                mask_tr = cat([tensor([1 for _ in range(sizes[0])]),
                               tensor([0 for _ in range(sizes[1] + sizes[2])])]).bool()
                mask_val = cat([tensor([0 for _ in range(sizes[0])]),
                                tensor([1 for _ in range(sizes[1])]),
                                tensor([0 for _ in range(sizes[2])])]).bool()
                mask_te = ~(mask_tr | mask_val)
            if geo is not None:
                return (KnowledgeGraph(
                            kg={'heads': self.head_idx[mask_tr],
                                'tails': self.tail_idx[mask_tr],
                                'relations': self.relations[mask_tr],
                                'point':self.point[mask_tr]},
                            ent2ix=self.ent2ix, rel2ix=self.rel2ix,
                            dict_of_heads=self.dict_of_heads,
                            dict_of_tails=self.dict_of_tails,
                            dict_of_rel=self.dict_of_rel,
                            id2point=self.id2point, geo=geo),
                        KnowledgeGraph(
                            kg={'heads': self.head_idx[mask_val],
                                'tails': self.tail_idx[mask_val],
                                'relations': self.relations[mask_val],
                                'point':self.point[mask_val]},
                            ent2ix=self.ent2ix, rel2ix=self.rel2ix,
                            dict_of_heads=self.dict_of_heads,
                            dict_of_tails=self.dict_of_tails,
                            dict_of_rel=self.dict_of_rel,
                            id2point=self.id2point, geo=geo),
                        KnowledgeGraph(
                            kg={'heads': self.head_idx[mask_te],
                                'tails': self.tail_idx[mask_te],
                                'relations': self.relations[mask_te],
                                'point':self.point[mask_te]},
                            ent2ix=self.ent2ix, rel2ix=self.rel2ix,
                            dict_of_heads=self.dict_of_heads,
                            dict_of_tails=self.dict_of_tails,
                            dict_of_rel=self.dict_of_rel,
                            id2point=self.id2point, geo=geo))
            else:
                return (KnowledgeGraph(
                    kg={'heads': self.head_idx[mask_tr],
                        'tails': self.tail_idx[mask_tr],
                        'relations': self.relations[mask_tr]},
                    ent2ix=self.ent2ix, rel2ix=self.rel2ix,
                    dict_of_heads=self.dict_of_heads,
                    dict_of_tails=self.dict_of_tails,
                    dict_of_rel=self.dict_of_rel),
                        KnowledgeGraph(
                            kg={'heads': self.head_idx[mask_val],
                                'tails': self.tail_idx[mask_val],
                                'relations': self.relations[mask_val]},
                            ent2ix=self.ent2ix, rel2ix=self.rel2ix,
                            dict_of_heads=self.dict_of_heads,
                            dict_of_tails=self.dict_of_tails,
                            dict_of_rel=self.dict_of_rel),
                        KnowledgeGraph(
                            kg={'heads': self.head_idx[mask_te],
                                'tails': self.tail_idx[mask_te],
                                'relations': self.relations[mask_te]},
                            ent2ix=self.ent2ix, rel2ix=self.rel2ix,
                            dict_of_heads=self.dict_of_heads,
                            dict_of_tails=self.dict_of_tails,
                            dict_of_rel=self.dict_of_rel))
        else:
            # return training and testing graphs

            assert (((sizes is not None) and len(sizes) == 2) or
                    ((sizes is None) and not validation))
            if sizes is None:
                mask_tr, mask_te = self.get_mask(share, validation=False)
            else:
                mask_tr = cat([tensor([1 for _ in range(sizes[0])]),
                               tensor([0 for _ in range(sizes[1])])]).bool()
                mask_te = ~mask_tr
            return (KnowledgeGraph(
                        kg={'heads': self.head_idx[mask_tr],
                            'tails': self.tail_idx[mask_tr],
                            'relations': self.relations[mask_tr]},
                        ent2ix=self.ent2ix, rel2ix=self.rel2ix,
                        dict_of_heads=self.dict_of_heads,
                        dict_of_tails=self.dict_of_tails,
                        dict_of_rel=self.dict_of_rel),
                    KnowledgeGraph(
                        kg={'heads': self.head_idx[mask_te],
                            'tails': self.tail_idx[mask_te],
                            'relations': self.relations[mask_te]},
                        ent2ix=self.ent2ix, rel2ix=self.rel2ix,
                        dict_of_heads=self.dict_of_heads,
                    dict_of_tails=self.dict_of_tails,
                    dict_of_rel=self.dict_of_rel))

    def get_mask(self, share, validation=False):
        """Returns masks to split knowledge graph into train, test and
        optionally validation sets. The mask is first created by dividing
        samples between subsets based on relation equilibrium. Then if any
        entity is not present in the training subset it is manually added by
        assigning a share of the sample involving the missing entity either
        as head or tail.

        Parameters
        ----------
        share: float
        validation: bool

        Returns
        -------
        mask: torch.Tensor, shape: (n), dtype: torch.bool
        mask_val: torch.Tensor, shape: (n), dtype: torch.bool (optional)
        mask_te: torch.Tensor, shape: (n), dtype: torch.bool
        """

        uniques_r, counts_r = self.relations.unique(return_counts=True)
        uniques_e, _ = cat((self.head_idx,
                            self.tail_idx)).unique(return_counts=True)

        mask = zeros_like(self.relations).bool()
        if validation:
            mask_val = zeros_like(self.relations).bool()

        # splitting relations among subsets
        for i, r in enumerate(uniques_r):
            rand = randperm(counts_r[i].item())

            # list of indices k such that relations[k] == r
            sub_mask = eq(self.relations, r).nonzero(as_tuple=False)[:, 0]

            assert len(sub_mask) == counts_r[i].item()

            if validation:
                train_size, val_size, test_size = self.get_sizes(counts_r[i].item(),
                                                                 share=share,
                                                                 validation=True)
                mask[sub_mask[rand[:train_size]]] = True
                mask_val[sub_mask[rand[train_size:train_size + val_size]]] = True

            else:
                train_size, test_size = self.get_sizes(counts_r[i].item(),
                                                       share=share,
                                                       validation=False)
                mask[sub_mask[rand[:train_size]]] = True

        # adding missing entities to the train set
        u = cat((self.head_idx[mask], self.tail_idx[mask])).unique()
        if len(u) < self.n_ent:
            missing_entities = tensor(list(set(uniques_e.tolist()) -
                                           set(u.tolist())), dtype=long)
            for e in missing_entities:
                sub_mask = ((self.head_idx == e) |
                            (self.tail_idx == e)).nonzero(as_tuple=False)[:, 0]
                rand = randperm(len(sub_mask))
                sizes = self.get_sizes(mask.shape[0],
                                       share=share,
                                       validation=validation)
                mask[sub_mask[rand[:sizes[0]]]] = True
                if validation:
                    mask_val[sub_mask[rand[:sizes[0]]]] = False

        if validation:
            assert not (mask & mask_val).any().item()
            return mask, mask_val, ~(mask | mask_val)
        else:
            return mask, ~mask

    @staticmethod
    def get_sizes(count, share, validation=False):
        """With `count` samples, returns how many should go to train and test

        """
        if count == 1:
            if validation:
                return 1, 0, 0
            else:
                return 1, 0
        if count == 2:
            if validation:
                return 1, 1, 0
            else:
                return 1, 1

        n_train = int(count * share)
        assert n_train < count
        if n_train == 0:
            n_train += 1

        if not validation:
            return n_train, count - n_train
        else:
            if count - n_train == 1:
                n_train -= 1
                return n_train, 1, 1
            else:
                n_val = int(int(count - n_train) / 2)
                return n_train, n_val, count - n_train - n_val

    def evaluate_dicts(self):
        """Evaluates dicts of possible alternatives to an entity in a fact
        that still gives a true fact in the entire knowledge graph.

        """
        for i in range(self.n_facts):
            self.dict_of_heads[(self.tail_idx[i].item(),
                                self.relations[i].item())].add(self.head_idx[i].item())
            self.dict_of_tails[(self.head_idx[i].item(),
                                self.relations[i].item())].add(self.tail_idx[i].item())
            self.dict_of_rel[(self.head_idx[i].item(),
                                self.tail_idx[i].item())].add(self.relations[i].item())


class SmallKG(Dataset):
    """Minimalist version of a knowledge graph. Built with tensors of heads,
    tails and relations.

    """
    def __init__(self, heads, tails, relations):
        assert heads.shape == tails.shape == relations.shape
        self.head_idx = heads
        self.tail_idx = tails
        self.relations = relations
        self.length = heads.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        return self.head_idx[item].item(), self.tail_idx[item].item(), self.relations[item].item()
