from importlib import import_module
import pandas as pd


def main():
    benchmarks = 'GeoDBpedia21'
    load_data = getattr(import_module('torchkge.utils.datasets'), 'load_'+benchmarks)
    print('Data process...')
    kg_train, kg_val, kg_test = load_data()
    print(f'Train set: {kg_train.n_ent} entities, {kg_train.n_rel} relations, {kg_train.n_facts} triplets.')
    print(f'Valid set: {kg_val.n_facts} triplets, Test set: {kg_test.n_facts} triplets.')
    train2id = pd.DataFrame({'head':kg_train.head_idx, 'rel':kg_train.relations, 'tail':kg_train.tail_idx})
    test2id = pd.DataFrame({'head':kg_test.head_idx, 'rel':kg_test.relations, 'tail':kg_test.tail_idx})
    valid2id = pd.DataFrame({'head':kg_val.head_idx, 'rel':kg_val.relations, 'tail':kg_val.tail_idx})

    ent2id = pd.DataFrame({'ent':kg_train.ent2ix.keys(), 'idx':kg_train.ent2ix.values()})
    rel2id = pd.DataFrame({'rel':kg_train.rel2ix.keys(), 'idx':kg_train.rel2ix.values()})
    id2ent = dict(zip(ent2id['idx'], ent2id['ent']))
    id2rel = dict(zip(rel2id['idx'], rel2id['rel']))
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



if __name__ == '__main__':
    main()