from torch import cuda
from torchkge.evaluation import LinkPredictionEvaluator, RelationPredictionEvaluator
import os
from importlib import import_module
from torchkge.utils.my_utils import load_ckpt, save_ckpt, create_dir_not_exists,time_since
import time

def val_eval(model, optimizer, model_save_path, epoch, save_time, save_time_freq, best_score, eval_b_size, last_improve):
    if (time.time() - save_time) / 60 > save_time_freq:
        create_dir_not_exists('./checkpoint')
        model.eval()
        evaluator = LinkPredictionEvaluator(model, kg_val)
        evaluator.evaluate(b_size=eval_b_size, verbose=False)
        _, hit_at_k = evaluator.hit_at_k(10)  # val filter hit_k
        if hit_at_k > best_score:
            save_ckpt(model, optimizer, epoch, best_score, model_save_path)
            best_score = hit_at_k
            improve = '*'  # 在有提升的结果后面加上*标注
            last_improve = time.time()  # 验证集hit_k增大即认为有提升
        else:
            improve = ''
        save_time = time.time()
        msg = ', Val Hit@10: {:>5.2%} {}'
        print(msg.format(hit_at_k, improve))
        return best_score, last_improve, save_time


def test_eval(benchmarks, model_name, opt_method, GDR=False, emb_dim=100, eval_b_size=256):

    ent_dim = emb_dim
    rel_dim = emb_dim

    model_save_path = './checkpoint/' + benchmarks + '_' + model_name + '_' + opt_method + '.ckpt'  # 保存最佳hits k (ent)模型
    device = 'cuda:0' if cuda.is_available() else 'cpu'

    # Load dataset
    module = getattr(import_module('torchkge.models'), model_name+'Model')
    load_data = getattr(import_module('torchkge.utils.datasets'), 'load_'+benchmarks)

    print('Loading data...')
    kg_train, kg_val, kg_test = load_data(GDR=GDR)
    print(f'Train set: {kg_train.n_ent} entities, {kg_train.n_rel} relations, {kg_train.n_facts} triplets.')
    print(f'Valid set: {kg_val.n_facts} triplets, Test set: {kg_test.n_facts} triplets.')

    # # Define the model and criterion
    if 'TransE' in model_name:
        model = module(emb_dim, kg_train.n_ent, kg_train.n_rel, dissimilarity_type='L2')
    else:
        model = module(ent_dim, rel_dim, kg_train.n_ent, kg_train.n_rel)
    # Move everything to CUDA if available
    if device == 'cuda:0':
        cuda.empty_cache()
        model.to(device)

    if os.path.exists(model_save_path):  # 存在则加载模型 进行测试
        load_ckpt(model_save_path, model, train=False)
        print(f'loading ckpt successful, start evaluate on test data...')
        print(model)
        model.eval()
        lp_evaluator = LinkPredictionEvaluator(model, kg_test)
        lp_evaluator.evaluate(eval_b_size, verbose=True)
        lp_evaluator.print_results()
        rp_evaluator = RelationPredictionEvaluator(model, kg_test)
        rp_evaluator.evaluate(eval_b_size, verbose=True)
        rp_evaluator.print_results()
    else:
        print('No pretrain model found!')
    # Testing the best checkpoint on test dataset

if __name__ == "__main__":
    # Define some hyper-parameters for training
    # global optimizer
    benchmarks = 'Sweden'
    model_name = 'TransE'
    opt_method = 'Adam'   # "Adagrad" "Adadelta" "Adam" "SGD"
    GDR = False  # 是否引入坐标信息 (引入坐标信息后测评会出现mean dis)
    emb_dim = 100  # TransE model
    eval_b_size = 256  # 测评valid test 时batch size

    test_eval(benchmarks, model_name,opt_method, GDR, emb_dim, eval_b_size)

