# -*- coding: utf-8 -*-
import numpy as np
import datetime
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import argparse
import os
import pickle

from GCL4SR.datasets.build_witg import build_WITG_from_trainset
from dataset import GCL4SRData
from trainer import GCL4SR_Train
from model import GCL4SR
from utils import check_path, set_seed, EarlyStopping, get_matrix_and_num


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default='GCL4SR', type=str)
    parser.add_argument("--data_name", default='home', type=str)
    parser.add_argument("--data_dir", default='./datasets/home/', type=str)
    parser.add_argument("--output_dir", default='output/', type=str)
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    # optimizer
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--lr_dc", type=float, default=0.7, help='learning rate decay.')
    parser.add_argument("--lr_dc_step", type=int, default=5,
                        help='the number of steps after which the learning rate decay.')
    parser.add_argument("--weight_decay", type=float, default=5e-5, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")

    # transformer
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--num_attention_heads", default=2, type=int, help="number of heads")
    parser.add_argument("--hidden_act", default="gelu", type=str, help="activation function")
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout")

    # train args
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="number of batch_size")
    parser.add_argument("--max_seq_length", default=50, type=int, help="max sequence length")
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of model")
    parser.add_argument("--seed", default=2022, type=int, help="seed")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--patience", default=10, type=int, help="early stop patience")

    # graph neural network
    parser.add_argument("--gnn_dropout_prob", type=float, default=0.5, help="gnn dropout")
    parser.add_argument("--use_renorm", type=bool, default=True, help="use re-normalize when build witg")
    parser.add_argument("--use_scale", type=bool, default=False, help="use scale when build witg")
    parser.add_argument("--fast_run", type=bool, default=True, help="can reduce training time and memory")
    parser.add_argument("--sample_size", default=[20, 20], type=list, help='gnn sample')
    parser.add_argument("--lam1", type=float, default=1, help="loss lambda 1")
    parser.add_argument("--lam2", type=float, default=0.1, help="loss lambda 2")

    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)

    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    args.data_file = args.data_dir + args.data_name + '.txt'
    user_num, item_num, valid_rating_matrix, test_rating_matrix = \
        get_matrix_and_num(args.data_file)

    train_data = pickle.load(open(args.data_dir + 'train.pkl', 'rb'))
    valid_data = pickle.load(open(args.data_dir + 'valid.pkl', 'rb'))
    test_data = pickle.load(open(args.data_dir + 'test.pkl', 'rb'))

    args.item_size = item_num
    args.user_size = user_num

    # save model args
    args_str = f'{args.model_name}-{args.data_name}-{args.sample_size}'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    print(args)
    with open(args.log_file, 'a') as f:
        f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n')
        f.write(str(args) + '\n')

    args.train_matrix = valid_rating_matrix

    # save model
    checkpoint = args_str + '.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)
    try:
        global_graph = torch.load(args.data_dir + 'witg.pt')
    except:
        build_WITG_from_trainset(datapath=args.data_dir)
        global_graph = torch.load(args.data_dir + 'witg.pt')
    model = GCL4SR(args=args, global_graph=global_graph)

    train_dataset = GCL4SRData(args, train_data)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, pin_memory=True, num_workers=8)

    eval_dataset = GCL4SRData(args, valid_data)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    test_dataset = GCL4SRData(args, test_data)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    trainer = GCL4SR_Train(model, args)

    if args.do_eval:
        trainer.load(args.checkpoint_path)
        print(f'Load model from {args.checkpoint_path} for test!')
        scores, result_info = trainer.eval_stage(0, test_dataloader, full_sort=True, test=True)

    else:
        early_stopping = EarlyStopping(args.checkpoint_path, patience=args.patience, verbose=True)
        for epoch in range(args.epochs):
            trainer.train_stage(epoch, train_dataloader)
            scores, _ = trainer.eval_stage(epoch, eval_dataloader, full_sort=True, test=False)
            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        trainer.args.train_matrix = test_rating_matrix
        print('---------------Change to test_rating_matrix!-------------------')
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.eval_stage(0, test_dataloader, full_sort=True, test=True)

    print(args_str)
    print(result_info)
    with open(args.log_file, 'a') as f:
        f.write(args_str + '\n')
        f.write(result_info + '\n')


main()

