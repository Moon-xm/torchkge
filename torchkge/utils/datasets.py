# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>

This module's code is freely adapted from Scikit-Learn's
sklearn.datasets.base.py code.
"""

import shutil
import tarfile
import zipfile

from os import makedirs, remove
from os.path import exists
from pandas import concat, DataFrame, merge, read_csv
from urllib.request import urlretrieve

from torchkge.data_structures import KnowledgeGraph

from torchkge.utils import get_data_home
import pandas as pd
import numpy as np


def load_GADM9(data_home=None):
    """

    Parameters
    ----------
    data_home: str, optional
        Path to the `torchkge_data` directory (containing data folders). If
        files are not present on disk in this directory, they are downloaded
        and then placed in the right place.

    Returns
    -------
    kg_train: torchkge.data_structures.KnowledgeGraph
    kg_val: torchkge.data_structures.KnowledgeGraph
    kg_test: torchkge.data_structures.KnowledgeGraph

    """
    if data_home is None:
        data_home = get_data_home()
    data_path = data_home + '/GADM9'
    if exists(data_path + '/train.txt') and exists(data_path + '/test.txt') and exists(data_path + '/valid.txt'):
        df1 = read_csv(data_path + '/train.txt',
                       sep='\t', header=None, names=['from', 'rel', 'to'])
        df2 = read_csv(data_path + '/valid.txt',
                       sep='\t', header=None, names=['from', 'rel', 'to'])
        df3 = read_csv(data_path + '/test.txt',
                       sep='\t', header=None, names=['from', 'rel', 'to'])
        df = concat([df1, df2, df3])
        kg = KnowledgeGraph(df)
        return kg.split_kg(sizes=(len(df1), len(df2), len(df3)))
    else:
        df = read_csv(data_path + '/triples_all.txt',
                       sep='\t', header=None, names=['from', 'rel', 'to'],encoding='utf-8')
        kg = KnowledgeGraph(df)
        kg_train, kg_val, kg_test = kg.split_kg(share=0.8, validation=True)
        data_save('/GADM9',kg_train, kg_val, kg_test)
        return kg_train, kg_val, kg_test

def load_GeoDBpedia21(data_home=None, GDR=False):
    """


    Parameters
    ----------
    data_home: str, optional
        Path to the `torchkge_data` directory (containing data folders). If
        files are not present on disk in this directory, they are downloaded
        and then placed in the right place.

    Returns
    -------
    kg_train: torchkge.data_structures.KnowledgeGraph
    kg_val: torchkge.data_structures.KnowledgeGraph
    kg_test: torchkge.data_structures.KnowledgeGraph

    """
    if data_home is None:
        data_home = get_data_home()
    data_path = data_home + '/GeoDBpedia21'
    if GDR == True:
        geo = data_path + '/ent2point.txt'
    else:
        geo = None

    if exists(data_path + '/train.txt') and exists(data_path + '/test.txt') and exists(data_path + '/valid.txt'):
        df1 = read_csv(data_path + '/train.txt',
                       sep='\t', header=None, names=['from', 'rel', 'to'])
        df2 = read_csv(data_path + '/valid.txt',
                       sep='\t', header=None, names=['from', 'rel', 'to'])
        df3 = read_csv(data_path + '/test.txt',
                       sep='\t', header=None, names=['from', 'rel', 'to'])
        df = concat([df1, df2, df3])
        kg = KnowledgeGraph(df=df,geo=geo)
        kg_train, kg_val, kg_test = kg.split_kg(sizes=(len(df1), len(df2), len(df3)),geo=geo)
        return kg_train, kg_val, kg_test
    else:
        df = read_csv(data_path + '/my_experiment.txt',
                       sep='\t', header=None, names=['from', 'rel', 'to'],encoding='utf-8')
        kg = KnowledgeGraph(df=df,geo=geo)
        kg_train, kg_val, kg_test = kg.split_kg(share=0.8, validation=True,geo=geo)
        data_save('/GeoDBpedia21',kg_train, kg_val, kg_test)
        return kg_train, kg_val, kg_test

def load_fb13(data_home=None):
    """Load FB13 dataset.

    Parameters
    ----------
    data_home: str, optional
        Path to the `torchkge_data` directory (containing data folders). If
        files are not present on disk in this directory, they are downloaded
        and then placed in the right place.

    Returns
    -------
    kg_train: torchkge.data_structures.KnowledgeGraph
    kg_val: torchkge.data_structures.KnowledgeGraph
    kg_test: torchkge.data_structures.KnowledgeGraph

    """
    if data_home is None:
        data_home = get_data_home()
    data_path = data_home + '/FB13'
    if not exists(data_path):
        makedirs(data_path, exist_ok=True)
        urlretrieve("https://graphs.telecom-paristech.fr/data/torchkge/kgs/FB13.zip",
                    data_home + '/FB13.zip')
        with zipfile.ZipFile(data_home + '/FB13.zip', 'r') as zip_ref:
            zip_ref.extractall(data_home)
        remove(data_home + '/FB13.zip')

    df1 = read_csv(data_path + '/train2id.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to'])
    df2 = read_csv(data_path + '/valid2id.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to'])
    df3 = read_csv(data_path + '/test2id.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to'])
    df = concat([df1, df2, df3])
    kg = KnowledgeGraph(df)

    return kg.split_kg(sizes=(len(df1), len(df2), len(df3)))


def load_fb15k(data_home=None):
    """Load FB15k dataset. See `here
    <https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data>`__
    for paper by Bordes et al. originally presenting the dataset.

    Parameters
    ----------
    data_home: str, optional
        Path to the `torchkge_data` directory (containing data folders). If
        files are not present on disk in this directory, they are downloaded
        and then placed in the right place.

    Returns
    -------
    kg_train: torchkge.data_structures.KnowledgeGraph
    kg_val: torchkge.data_structures.KnowledgeGraph
    kg_test: torchkge.data_structures.KnowledgeGraph

    """
    if data_home is None:
        data_home = get_data_home()
    data_path = data_home + '/FB15k'
    if not exists(data_path):
        makedirs(data_path, exist_ok=True)
        urlretrieve("https://graphs.telecom-paristech.fr/data/torchkge/kgs/FB15k.zip",
                    data_home + '/FB15k.zip')
        with zipfile.ZipFile(data_home + '/FB15k.zip', 'r') as zip_ref:
            zip_ref.extractall(data_home)
        remove(data_home + '/FB15k.zip')

    df1 = read_csv(data_path + '/freebase_mtr100_mte100-train.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to'])
    df2 = read_csv(data_path + '/freebase_mtr100_mte100-valid.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to'])
    df3 = read_csv(data_path + '/freebase_mtr100_mte100-test.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to'])
    df = concat([df1, df2, df3])
    kg = KnowledgeGraph(df)

    return kg.split_kg(sizes=(len(df1), len(df2), len(df3)))


def load_fb15k237(data_home=None):
    """Load FB15k237 dataset. See `here
    <https://www.aclweb.org/anthology/D15-1174/>`__ for paper by Toutanova et
    al. originally presenting the dataset.

    Parameters
    ----------
    data_home: str, optional
        Path to the `torchkge_data` directory (containing data folders). If
        files are not present on disk in this directory, they are downloaded
        and then placed in the right place.

    Returns
    -------
    kg_train: torchkge.data_structures.KnowledgeGraph
    kg_val: torchkge.data_structures.KnowledgeGraph
    kg_test: torchkge.data_structures.KnowledgeGraph

    """
    if data_home is None:
        data_home = get_data_home()
    data_path = data_home + '/FB15k237'
    if not exists(data_path):
        makedirs(data_path, exist_ok=True)
        urlretrieve("https://graphs.telecom-paristech.fr/data/torchkge/kgs/FB15k237.zip",
                    data_home + '/FB15k237.zip')
        with zipfile.ZipFile(data_home + '/FB15k237.zip', 'r') as zip_ref:
            zip_ref.extractall(data_home)
        remove(data_home + '/FB15k237.zip')

    df1 = read_csv(data_path + '/train.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to'])
    df2 = read_csv(data_path + '/valid.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to'])
    df3 = read_csv(data_path + '/test.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to'])
    df = concat([df1, df2, df3])
    kg = KnowledgeGraph(df)

    return kg.split_kg(sizes=(len(df1), len(df2), len(df3)))


def load_wn18(data_home=None):
    """Load WN18 dataset.

    Parameters
    ----------
    data_home: str, optional
        Path to the `torchkge_data` directory (containing data folders). If
        files are not present on disk in this directory, they are downloaded
        and then placed in the right place.

    Returns
    -------
    kg_train: torchkge.data_structures.KnowledgeGraph
    kg_val: torchkge.data_structures.KnowledgeGraph
    kg_test: torchkge.data_structures.KnowledgeGraph

    """
    if data_home is None:
        data_home = get_data_home()
    data_path = data_home + '/WN18'
    if not exists(data_path):
        makedirs(data_path, exist_ok=True)
        urlretrieve("https://graphs.telecom-paristech.fr/data/torchkge/kgs/WN18.zip",
                    data_home + '/WN18.zip')
        with zipfile.ZipFile(data_home + '/WN18.zip', 'r') as zip_ref:
            zip_ref.extractall(data_home)
        remove(data_home + '/WN18.zip')

    df1 = read_csv(data_path + '/wordnet-mlj12-train.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to'])
    df2 = read_csv(data_path + '/wordnet-mlj12-valid.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to'])
    df3 = read_csv(data_path + '/wordnet-mlj12-test.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to'])
    df = concat([df1, df2, df3])
    kg = KnowledgeGraph(df)

    return kg.split_kg(sizes=(len(df1), len(df2), len(df3)))


def load_wn18rr(data_home=None):
    """Load WN18RR dataset. See `here
    <https://arxiv.org/abs/1707.01476>`__ for paper by Dettmers et
    al. originally presenting the dataset.

    Parameters
    ----------
    data_home: str, optional
        Path to the `torchkge_data` directory (containing data folders). If
        files are not present on disk in this directory, they are downloaded
        and then placed in the right place.

    Returns
    -------
    kg_train: torchkge.data_structures.KnowledgeGraph
    kg_val: torchkge.data_structures.KnowledgeGraph
    kg_test: torchkge.data_structures.KnowledgeGraph

    """
    if data_home is None:
        data_home = get_data_home()
    data_path = data_home + '/WN18RR'
    if not exists(data_path):
        makedirs(data_path, exist_ok=True)
        urlretrieve("https://graphs.telecom-paristech.fr/data/torchkge/kgs/WN18RR.zip",
                    data_home + '/WN18RR.zip')
        with zipfile.ZipFile(data_home + '/WN18RR.zip', 'r') as zip_ref:
            zip_ref.extractall(data_home)
        remove(data_home + '/WN18RR.zip')

    df1 = read_csv(data_path + '/train.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to'])
    df2 = read_csv(data_path + '/valid.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to'])
    df3 = read_csv(data_path + '/test.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to'])
    df = concat([df1, df2, df3])
    kg = KnowledgeGraph(df)

    return kg.split_kg(sizes=(len(df1), len(df2), len(df3)))


def load_yago3_10(data_home=None):
    """Load YAGO3-10 dataset. See `here
    <https://arxiv.org/abs/1707.01476>`__ for paper by Dettmers et
    al. originally presenting the dataset.

    Parameters
    ----------
    data_home: str, optional
        Path to the `torchkge_data` directory (containing data folders). If
        files are not present on disk in this directory, they are downloaded
        and then placed in the right place.

    Returns
    -------
    kg_train: torchkge.data_structures.KnowledgeGraph
    kg_val: torchkge.data_structures.KnowledgeGraph
    kg_test: torchkge.data_structures.KnowledgeGraph

    """
    if data_home is None:
        data_home = get_data_home()
    data_path = data_home + '/YAGO3-10'
    if not exists(data_path):
        makedirs(data_path, exist_ok=True)
        urlretrieve("https://graphs.telecom-paristech.fr/data/torchkge/kgs/YAGO3-10.zip",
                    data_home + '/YAGO3-10.zip')
        with zipfile.ZipFile(data_home + '/YAGO3-10.zip', 'r') as zip_ref:
            zip_ref.extractall(data_home)
        remove(data_home + '/YAGO3-10.zip')

    df1 = read_csv(data_path + '/train.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to'])
    df2 = read_csv(data_path + '/valid.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to'])
    df3 = read_csv(data_path + '/test.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to'])
    df = concat([df1, df2, df3])
    kg = KnowledgeGraph(df)

    return kg.split_kg(sizes=(len(df1), len(df2), len(df3)))


def load_wikidatasets(which, limit_=0, data_home=None):
    """Load WikiDataSets dataset. See `here
    <https://arxiv.org/abs/1906.04536>`__ for paper by Boschin et al.
    originally presenting the dataset.

    Parameters
    ----------
    which: str
        String indicating which subset of Wikidata should be loaded.
        Available ones are `humans`, `companies`, `animals`, `countries` and
        `films`.
    limit_: int, optional (default=0)
        This indicates a lower limit on the number of neighbors an entity
        should have in the graph to be kept.
    data_home: str, optional
        Path to the `torchkge_data` directory (containing data folders). If
        files are not present on disk in this directory, they are downloaded
        and then placed in the right place.

    Returns
    -------
    kg: torchkge.data_structures.KnowledgeGraph

    """
    assert which in ['humans', 'companies', 'animals', 'countries', 'films']

    if data_home is None:
        data_home = get_data_home()

    data_home = data_home + '/WikiDataSets'
    data_path = data_home + '/' + which
    if not exists(data_path):
        makedirs(data_path, exist_ok=True)
        urlretrieve("https://graphs.telecom-paristech.fr/data/WikiDataSets/{}.tar.gz".format(which),
                    data_home + '/{}.tar.gz'.format(which))

        with tarfile.open(data_home + '/{}.tar.gz'.format(which), 'r') as tf:
            tf.extractall(data_home)
        remove(data_home + '/{}.tar.gz'.format(which))

    df = read_csv(data_path + '/edges.tsv', sep='\t',
                  names=['from', 'to', 'rel'], skiprows=1)

    if limit_ > 0:
        a = df.groupby('from').count()['rel']
        b = df.groupby('to').count()['rel']

        # Filter out nodes with too few facts
        tmp = merge(right=DataFrame(a).reset_index(),
                    left=DataFrame(b).reset_index(),
                    how='outer', right_on='from', left_on='to', ).fillna(0)

        tmp['rel'] = tmp['rel_x'] + tmp['rel_y']
        tmp = tmp.drop(['from', 'rel_x', 'rel_y'], axis=1)

        tmp = tmp.loc[tmp['rel'] >= limit_]
        df_bis = df.loc[df['from'].isin(tmp['to']) | df['to'].isin(tmp['to'])]

        kg = KnowledgeGraph(df_bis)
    else:
        kg = KnowledgeGraph(df)

    return kg


def load_wikidata_vitals(level=5, data_home=None):
    """Load knowledge graph extracted from Wikidata using the entities
    corresponding to Wikipedia pages contained in Wikivitals. See `here
    <https://netset.telecom-paris.fr/>`__ for details on Wikivitals and
    Wikivitals+ datasets.

    Parameters
    ----------
    level: int (default=5)
        Either 4 or 5.
    data_home: str, optional
        Path to the `torchkge_data` directory (containing data folders). If
        files are not present on disk in this directory, they are downloaded
        and then placed in the right place.

    Returns
    -------
    kg: torchkge.data_structures.KnowledgeGraph
    kg_attr: torchkge.data_structures.KnowledgeGraph
    """
    assert level in [4, 5]

    if data_home is None:
        data_home = get_data_home()

    data_path = data_home + '/wikidatavitals-level{}'.format(level)

    if not exists(data_path):
        makedirs(data_path, exist_ok=True)
        urlretrieve("https://graphs.telecom-paristech.fr/data/torchkge/kgs/wikidatavitals-level{}.zip".format(level),
                    data_home + '/wikidatavitals-level{}.zip'.format(level))

        with zipfile.ZipFile(data_home + '/wikidatavitals-level{}.zip'.format(level), 'r') as zip_ref:
            zip_ref.extractall(data_home)
        remove(data_home + '/wikidatavitals-level{}.zip'.format(level))

    df = read_csv(data_path + '/edges.tsv', sep='\t',
                  names=['from', 'to', 'rel'], skiprows=1)
    attributes = read_csv(data_path + '/attributes.tsv', sep='\t',
                  names=['from', 'to', 'rel'], skiprows=1)

    entities = read_csv(data_path + '/entities.tsv', sep='\t')
    relations = read_csv(data_path + '/relations.tsv', sep='\t')
    nodes = read_csv(data_path + '/nodes.tsv', sep='\t')

    df = enrich(df, entities, relations)
    attributes = enrich(attributes, entities, relations)

    relid2label = {relations.loc[i, 'wikidataID']: relations.loc[i, 'label']
                   for i in relations.index}
    entid2label = {entities.loc[i, 'wikidataID']: entities.loc[i, 'label'] for
                   i in entities.index}
    entid2pagename = {nodes.loc[i, 'wikidataID']: nodes.loc[i, 'pageName'] for
                      i in nodes.index}

    kg = KnowledgeGraph(df)
    kg_attr = KnowledgeGraph(attributes)

    kg.relid2label = relid2label
    kg_attr.relid2label = relid2label
    kg.entid2label = entid2label
    kg_attr.entid2label = entid2label
    kg.entid2pagename = entid2pagename
    kg_attr.entid2pagename = entid2pagename

    return kg, kg_attr


def enrich(df, entities, relations):
    df = merge(left=df, right=entities[['entityID', 'wikidataID']],
               left_on='from', right_on='entityID')[
        ['to', 'rel', 'wikidataID']]
    df.columns = ['to', 'rel', 'from']

    df = merge(left=df, right=entities[['entityID', 'wikidataID']],
               left_on='to', right_on='entityID')[
        ['from', 'rel', 'wikidataID']]

    df.columns = ['from', 'rel', 'to']

    df = merge(left=df, right=relations[['relationID', 'wikidataID']],
               left_on='rel', right_on='relationID')[
        ['from', 'to', 'wikidataID']]

    df.columns = ['from', 'to', 'rel']
    return df

def id2point(h, t, idx2point):
    # h_point = h.map(lambda x: idx2point[h])
    # t_point = t.map(lambda x: idx2point[t])
    # for (h_i, t_i) in zip(h, t):
    #     pass
    point = np.array([[idx2point[h_i.item()], idx2point[t_i.item()]] for (h_i, t_i) in zip(h, t)])
    return point



def data_save(benchmarks,kg_train, kg_val, kg_test):
    print('Split the data...')
    # print(f'Train set: {kg_train.n_ent} entities, {kg_train.n_rel} relations, {kg_train.n_facts} triplets.')
    # print(f'Valid set: {kg_val.n_facts} triplets, Test set: {kg_test.n_facts} triplets.')
    train2id = pd.DataFrame({'head':kg_train.head_idx, 'rel':kg_train.relations, 'tail':kg_train.tail_idx})
    test2id = pd.DataFrame({'head':kg_test.head_idx, 'rel':kg_test.relations, 'tail':kg_test.tail_idx})
    valid2id = pd.DataFrame({'head':kg_val.head_idx, 'rel':kg_val.relations, 'tail':kg_val.tail_idx})
    ent2id = pd.DataFrame({'ent':kg_train.ent2ix.keys(), 'idx':kg_train.ent2ix.values()})
    rel2id = pd.DataFrame({'rel':kg_train.rel2ix.keys(), 'idx':kg_train.rel2ix.values()})
    id2ent = dict(zip(ent2id['idx'], ent2id['ent']))
    id2rel = dict(zip(rel2id['idx'], rel2id['rel']))
    id2point = pd.DataFrame({'id':kg_train.id2point.keys(), 'point':kg_train.id2point.values()})
    id2point = id2point.sort_values(by='id', ascending=True)
    train = pd.DataFrame({'head':train2id['head'].map(lambda x: id2ent[x]), 'rel':train2id['rel'].map(lambda x: id2rel[x]), 'tail': train2id['tail'].map(lambda x: id2ent[x])})
    valid = pd.DataFrame({'head':valid2id['head'].map(lambda x: id2ent[x]), 'rel':valid2id['rel'].map(lambda x: id2rel[x]), 'tail': valid2id['tail'].map(lambda x: id2ent[x])})
    test = pd.DataFrame({'head':test2id['head'].map(lambda x: id2ent[x]), 'rel':test2id['rel'].map(lambda x: id2rel[x]), 'tail': test2id['tail'].map(lambda x: id2ent[x])})
    train.to_csv(path_or_buf='benchmarks/'+benchmarks+'/train.txt', sep='\t', header=False, index=False)
    valid.to_csv(path_or_buf='benchmarks/'+benchmarks+'/valid.txt', sep='\t', header=False, index=False)
    test.to_csv(path_or_buf='benchmarks/'+benchmarks+'/test.txt', sep='\t', header=False, index=False)
    train2id.to_csv(path_or_buf='benchmarks/'+benchmarks+'/train2id.txt', sep='\t', header=False, index=False)
    test2id.to_csv(path_or_buf='benchmarks/'+benchmarks+'/test2id.txt', sep='\t', header=False, index=False)
    valid2id.to_csv(path_or_buf='benchmarks/'+benchmarks+'/valid2id.txt', sep='\t', header=False, index=False)
    ent2id.to_csv(path_or_buf='benchmarks/'+benchmarks+'/ent2id.txt', sep='\t', header=False, index=False)
    rel2id.to_csv(path_or_buf='benchmarks/'+benchmarks+'/rel2id.txt', sep='\t', header=False, index=False)
    id2point.to_csv(path_or_buf='benchmarks/'+benchmarks+'/id2point.txt', sep='\t', header=False, index=False)
