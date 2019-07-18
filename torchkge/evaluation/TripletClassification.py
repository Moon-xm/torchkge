# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
aboschin@enst.fr
"""

from torchkge.sampling import PositionalNegativeSampler


class TripletClassificationEvaluator(object):
    """Evaluate performance of given embedding using triplet classification method.

    References
    ----------
    * Richard Socher, Danqi Chen, Christopher D Manning, and Andrew Ng.
      Reasoning With Neural Tensor Networks for Knowledge Base Completion.
      In Advances in Neural Information Processing Systems 26, pages 926–934. 2013.
      https://nlp.stanford.edu/pubs/SocherChenManningNg_NIPS2013.pdf
    """

    def __init__(self, model, kg_val, kg_test):
        """
        Parameters
        ----------
        model
        kg_val
        kg_test

        """
        self.model = model
        self.kg_val = kg_val
        self.kg_test = kg_test

        self.evaluated = False

        self.sampler = PositionalNegativeSampler(self.kg_val, kg_test=self.kg_test)

    def evaluate(self):
        """Find relation thresholds.

        """
        self.model.forward(self.kg_val.head_idx, self.kg_val.head_idx)

    def accuracy(self):
        pass
