from torch import cuda
from torch.optim import Adam

from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss, DataLoader
from torchkge.evaluation import LinkPredictionEvaluator
from torchkge.utils.optim import optimizer
from torchkge.utils.datasets import id2point

from torchkge.utils.my_utils import load_ckpt, save_ckpt, create_dir_not_exists,time_since

import os
from importlib import import_module
import time


def main():
    # Define some hyper-parameters for training
    global optimizer
    benchmarks = 'GeoDBpedia21'
    model_name = 'TransD'
    opt_method = 'Adam'   # "Adagrad" "Adadelta" "Adam" "SGD"
    GDR = False  # 是否引入坐标信息

    emb_dim = 100  # TransE model
    ent_dim = emb_dim
    rel_dim = emb_dim
    lr = 0.001
    margin = 1

    n_epochs = 20000
    train_b_size = 5120  # 训练时batch size
    eval_b_size = 256  # 测评valid test 时batch size
    validation_freq = 500  # 多少轮进行在验证集进行一次测试 同时保存最佳模型
    require_improvement = validation_freq*5  # 验证集top_k超过多少epoch没下降，结束训练
    model_save_path = './checkpoint/' + benchmarks + '_' + model_name + '.ckpt'  # 保存最佳hits k (ent)模型
    device = 'cuda:0' if cuda.is_available() else 'cpu'

    # Load dataset
    module = getattr(import_module('torchkge.models'), model_name+'Model')
    load_data = getattr(import_module('torchkge.utils.datasets'), 'load_'+benchmarks)

    print('Loading data...')
    kg_train, kg_val, kg_test = load_data(GDR=GDR)
    print(f'Train set: {kg_train.n_ent} entities, {kg_train.n_rel} relations, {kg_train.n_facts} triplets.')
    print(f'Valid set: {kg_val.n_facts} triplets, Test set: {kg_test.n_facts} triplets.')

    # Define the model and criterion
    print('Loading model...')
    if 'TransE' in model_name:
        model = module(emb_dim, kg_train.n_ent, kg_train.n_rel, dissimilarity_type='L2')
    else:
        model = module(ent_dim, rel_dim, kg_train.n_ent, kg_train.n_rel)
    criterion = MarginLoss(margin)

    # Move everything to CUDA if available
    if device == 'cuda:0':
        cuda.empty_cache()
        model.to(device)
        criterion.to(device)
        dataloader = DataLoader(kg_train, batch_size=train_b_size, use_cuda='all')
    else:
        dataloader = DataLoader(kg_train, batch_size=train_b_size, use_cuda=None)

    # Define the torch optimizer to be used
    optimizer = optimizer(model, opt_method=opt_method, lr=lr)
    # optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sampler = BernoulliNegativeSampler(kg_train)

    start_epoch = 1
    best_score = float('-inf')
    if os.path.exists(model_save_path):  # 存在则加载模型 并继续训练
        start_epoch, best_score = load_ckpt(model_save_path, model, optimizer)
        print(f'loading ckpt sucessful, start on epoch {start_epoch}...')
    print(model)
    print('lr: {}, margin: {}, dim {}, total epoch: {}, device: {}, batch size: {}, optim: {}'\
    .format(lr, margin, emb_dim, n_epochs, device, train_b_size, opt_method))

    print('Training...')
    last_improve = start_epoch  # 记录上次验证集loss下降的epoch数
    start = time.time()
    for epoch in range(start_epoch, n_epochs+1):
        running_loss = 0.0
        model.train()
        for i, batch in enumerate(dataloader):
            h, t, r = batch[0], batch[1], batch[2]
            n_h, n_t = sampler.corrupt_batch(h, t, r)  # 1:1 negative sampling
            # n_point = id2point(n_h, n_t, kg_train.id2point)
            optimizer.zero_grad()

            # forward + backward + optimize
            pos, neg = model(h, t, n_h, n_t, r)
            loss = criterion(pos, neg)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print('\rEpoch [{:>4}/{:>4}] | mean loss: {:>8.3f}, time: {}'.format(epoch, n_epochs, running_loss / len(dataloader), time_since(start)), end='', flush=True)
        # test
        if epoch % validation_freq == 0:
            create_dir_not_exists('./checkpoint')
            model.eval()
            evaluator = LinkPredictionEvaluator(model, kg_val)
            evaluator.evaluate(b_size=eval_b_size, verbose=False)
            _, hit_at_k = evaluator.hit_at_k(10)  # val filter hit_k
            if hit_at_k > best_score:
                save_ckpt(model, optimizer, epoch, best_score, model_save_path)
                best_score = hit_at_k
                improve = '*'  # 在有提升的结果后面加上*标注
                last_improve = epoch  # 验证集hit_k增大即认为有提升
            else:
                improve = ''
            msg = '\nTrain loss: {:>8.3f}, Val Hit@10: {:>5.2%}, Time {} {}'
            print(msg.format(running_loss / len(dataloader), hit_at_k, time_since(start), improve))
        model.normalize_parameters()
        if epoch - last_improve > require_improvement:
            # 验证集top_k超过一定epoch没增加，结束训练
            print("\nNo optimization for a long time, auto-stopping...")
            break

    print('\nTraining done, start evaluate on test data...')
    # Testing the best checkpoint on test dataset
    load_ckpt(model_save_path, model, optimizer)
    model.eval()
    evaluator = LinkPredictionEvaluator(model, kg_test)
    evaluator.evaluate(eval_b_size, verbose=False)
    evaluator.print_results()

if __name__ == "__main__":
    main()
