from importlib import import_module
import pandas as pd


def main():
    benchmarks = 'GADM9'
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
    train2id.to_csv(path_or_buf='benchmarks/'+benchmarks+'/train2id.txt', sep='\t', header=False, index=False)
    test2id.to_csv(path_or_buf='benchmarks/'+benchmarks+'/test2id.txt', sep='\t', header=False, index=False)
    valid2id.to_csv(path_or_buf='benchmarks/'+benchmarks+'/valid2id.txt', sep='\t', header=False, index=False)
    ent2id.to_csv(path_or_buf='benchmarks/'+benchmarks+'/ent2id.txt', sep='\t', header=False, index=False)
    rel2id.to_csv(path_or_buf='benchmarks/'+benchmarks+'/rel2id.txt', sep='\t', header=False, index=False)

if __name__ == '__main__':
    main()