from torch import cuda
from torch.optim import Adam

# from torchkge.models import TransEModel
from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss, DataLoader
# from torchkge.utils.datasets import load_GADM9
from torchkge.evaluation import LinkPredictionEvaluator

from torchkge.utils.my_utils import load_ckpt, save_ckpt, create_dir_not_exists

import os
from importlib import import_module
from tqdm.autonotebook import tqdm


def main():
    # Define some hyper-parameters for training
    model_name = 'TransE'
    benchmarks = 'GeoDBpedia21'
    emb_dim = 100
    lr = 0.0004
    n_epochs = 1000
    b_size = 5120
    margin = 0.5
    validation_freq = 20  # 多少轮进行在验证集进行一次测试
    model_save_path = './checkpoint/' + benchmarks + '/' + model_name + '.ckpt'  # 保存最佳hits k (ent)模型
    device = 'cuda:0' if cuda.is_available() else 'cpu'

    # Load dataset
    module = getattr(import_module('torchkge.models'), model_name+'Model')
    load_data = getattr(import_module('torchkge.utils.datasets'), 'load_'+benchmarks)

    print('Loading data...')
    kg_train, kg_val, kg_test = load_data()
    print(f'Train set: {kg_train.n_ent} entities, {kg_train.n_rel} relations, {kg_train.n_facts} triplets.')
    print(f'Valid set: {kg_val.n_facts} triplets, Test set: {kg_test.n_facts} triplets.')

    # Define the model and criterion
    print('Loading model...')
    model = module(emb_dim, kg_train.n_ent, kg_train.n_rel, dissimilarity_type='L2')
    # model = TransEModel(emb_dim, kg_train.n_ent, kg_train.n_rel, dissimilarity_type='L2')
    criterion = MarginLoss(margin)
    # Move everything to CUDA if available
    if device == 'cuda:0':
        cuda.empty_cache()
        model.to(device)
        criterion.to(device)
        dataloader = DataLoader(kg_train, batch_size=b_size, use_cuda='all')
    else:
        dataloader = DataLoader(kg_train, batch_size=b_size, use_cuda=None)


    # Define the torch optimizer to be used
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    sampler = BernoulliNegativeSampler(kg_train)

    start_epoch = 1
    best_score = 0.0
    if os.path.exists(model_save_path):  # 存在则加载模型 并继续训练
        start_epoch, best_score = load_ckpt(model_save_path, model, optimizer)
    print(model)
    print('lr: {}, margin: {}, dim {}, device: {}\n optim: {}, batch size: {}'.format(lr, margin, emb_dim,
                                                                                            device, optimizer,
                                                                                            b_size))

    iterator = tqdm(range(start_epoch, n_epochs), unit='epoch')
    for epoch in iterator:
        running_loss = 0.0
        model.train()
        for i, batch in enumerate(dataloader):
            h, t, r = batch[0], batch[1], batch[2]
            n_h, n_t = sampler.corrupt_batch(h, t, r)
            optimizer.zero_grad()

            # forward + backward + optimize
            pos, neg = model(h, t, n_h, n_t, r)
            loss = criterion(pos, neg)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        iterator.set_description(
            'Epoch {} | mean loss: {:.5f}'.format(epoch,
                                                  running_loss / len(dataloader)))
        # test
        if epoch % validation_freq == 0:
            create_dir_not_exists('./checkpoint')
            evaluator = LinkPredictionEvaluator(model, kg_val)
            evaluator.evaluate(b_size=256, verbose=False)
            val_mrr = evaluator.mrr()[1]
            if val_mrr > best_score:
                best_score = val_mrr
                improve = '*'  # 在有提升的结果后面加上*标注
                save_ckpt(model, optimizer, epoch, best_score, config.model_save_path)
            else:
                improve = ''
            msg = 'Epoch {} | Train loss: {}, Validation MRR: {} {}'
            print(msg.format(epoch, running_loss / len(dataloader), val_mrr, improve))
    model.normalize_parameters()

    print('Training done, start evaluate on test data...')
    # Testing the best checkpoint on test dataset
    load_ckpt(model_save_path, model, optimizer)
    model.eval()
    evaluator = LinkPredictionEvaluator(model, kg_test)
    evaluator.evaluate(200)
    evaluator.print_results()


if __name__ == "__main__":
    main()
