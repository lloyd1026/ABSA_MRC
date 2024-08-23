import os
import time

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertTokenizer, ElectraTokenizer, get_linear_schedule_with_warmup

from collate import collate_fn
from data_utils import ACOSDataset
from finetuning_argparse import init_args
from labels import get_aspect_category, get_category_sentiment_num_list
from mrc_model import MRCModel
from tools import get_logger, seed_everything, save_model, print_results, print_results2
from trainer import ACOSTrainer


def do_train():
    # ##########init model##########
    logger.info("Building test Model...")
    category_list = get_aspect_category(args.task, args.data_type)
    args.category_dim = len(category_list[0])

    # category and sentiment num_list
    res_lists = get_category_sentiment_num_list(args)
    args.category_num_list = res_lists[0]
    args.sentiment_num_list = res_lists[-1]

    # Base Model: Bert(Transformers),  使用PLM模型：sentiWSP    == fine_tuning ==> MRC_CLRI
    model = MRCModel(args, args.category_dim)
    model = model.to('cpu')

    # state_dict = model.state_dict()
    # for param_name in state_dict.keys():
    #     print(param_name)

    # dataset
    train_dataset = ACOSDataset(tokenizer, args, "train")   # train 训练集
    dev_dataset = ACOSDataset(tokenizer, args, "dev")       # development 验证集

    # dataloader
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True,
                                  collate_fn=collate_fn)
    # 训练的时候设置为args.eval_batch_size 若args.eval_batch_size过大，可能会出现cuda out of memory
    dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=1, collate_fn=collate_fn)

    # optimizer 优化器
    logger.info('initial optimizer......')
    # 获取模型中的所有参数的名称和值（即权重和偏置）
    param_optimizer = list(model.named_parameters())

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if "_bert" in n],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if "_bert" not in n],
         'lr': args.learning_rate1,
         'weight_decay': 0.01}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate2)  # 全局lr，默认

    # scheduler 学习率调度器
    batch_num_train = len(train_dataset) // args.train_batch_size
    training_steps = args.epoch_num * batch_num_train
    warmup_steps = int(training_steps * args.warm_up)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=training_steps)

    # trainer object
    trainer = ACOSTrainer(logger, model, optimizer, scheduler, tokenizer, args)

    # ##########Training##########
    logger.info("***** Running Training *****")
    # 初始化最佳性能指标
    best_dev_quadruple_f1, best_test_quadruple_f1, best_test_imp_quad_f1 = .0, .0, .0
    # 按训练周期(epoch)进行训练
    for epoch in range(1, args.epoch_num + 1):
        # Set the module in training mode.
        model.train()
        # TRAIN
        trainer.train(train_dataloader, epoch)
        # DEV    do_eval
        logger.info("***** Running Dev | Epoch {} *****".format(epoch))
        results = trainer.my_eval(dev_dataloader)

        if results['quintuple']['f1'] == 0:
            continue

        print_results(logger, results)
        if results['quintuple']['f1'] > best_dev_quadruple_f1:
            best_dev_quadruple_f1 = results['quintuple']['f1']
            save_path = save_model(output_path, f"{args.data_type}_test", epoch, optimizer, model)
            args.save_path = save_path
            logger.info("i got the best dev result {}...".format(best_dev_quadruple_f1))

    logger.info("***** Train Over *****")
    logger.info("The best dev quintuple f1: {}".format(best_dev_quadruple_f1))


# def do_test():
#     # ##########init model##########
#     logger.info("Building MRC-CLRI Model...")
#     category_list = get_aspect_category(args.task, args.data_type)
#     args.category_dim = len(category_list[0])
#     # category and sentiment num_list
#     res_lists = get_category_sentiment_num_list(args)
#     args.category_num_list = res_lists[0]
#     args.sentiment_num_list = res_lists[-1]
#
#     # # load data
#     test_dataset = ACOSDataset(tokenizer, args, "test")
#     # # dataloader
#     test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.eval_batch_size, collate_fn=collate_fn)
#     model = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
#     model = model.cuda()
#
#     trainer = ACOSTrainer(logger, model, None, None, tokenizer, args)
#     logger.info("***** Running Test *****")
#     # test_results = trainer.eval(test_dataloader)
#     test_results = trainer.batch_eval(test_dataloader)
#
#     logger.info(test_results)
#
#
# # ？？？？
# def do_eval(model, dev_dataloader, test_dataloader):
#
#     trainer = ACOSTrainer(logger, model, None, None, tokenizer, args)
#     # ##########Dev##########
#     logger.info("***** Running Dev *****")
#     dev_results = trainer.batch_eval(dev_dataloader)
#     if args.do_optimized:
#         print_results2(logger, dev_results)
#     else:
#         print_results(logger, dev_results)
#     logger.info("***** Running Test *****")
#     # test_results = trainer.eval(test_dataloader)
#     test_results = trainer.batch_eval(test_dataloader)
#     if args.do_optimized:
#         print_results2(logger, test_results)
#     else:
#         print_results(logger, test_results)
#
#     return dev_results, test_results


# ？？？？
# def do_optimized():
#     # ##########init model##########
#     logger.info("Building MRC-CLRI Model...")
#     category_list = get_aspect_category(args.task, args.data_type)
#     args.category_dim = len(category_list[0])
#     # category and sentiment num_list
#     res_lists = get_category_sentiment_num_list(args)
#     args.category_num_list = res_lists[0]
#     args.sentiment_num_list = res_lists[-1]
#
#     # load data
#     # dataset
#     dev_dataset = ACOSDataset(tokenizer, args, "dev")
#     test_dataset = ACOSDataset(tokenizer, args, "test")
#     # dataloader
#     dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=args.eval_batch_size, collate_fn=collate_fn)
#     test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.eval_batch_size, collate_fn=collate_fn)
#     model = torch.load(args.checkpoint_path)
#     model = model.cuda()
#
#     # 先确定beta再确定alpha(alpha=0.8)
#     start, end, step = args.alpha_start, args.alpha_end, args.alpha_step
#     alpha_list = [round(x, 2) for x in list(np.arange(start, end + step, step))]
#
#     start, end, step = args.beta_start, args.beta_end, args.beta_step
#     beta_list = [i for i in range(start, end + 1, step)]
#
#     dev_f1_list, test_f1_list = [], []
#     for b in beta_list:
#         args.beta = int(b)
#         logger.info(f'alpha is {args.alpha}, beta is {b}')
#         dev_results, test_results = do_eval(model, dev_dataloader, test_dataloader)
#
#         dev_f1_list.append(dev_results['quadruple']['f1'])
#         test_f1_list.append(test_results['quadruple']['f1'])
#     best_dev_f1, best_test_f1 = max(dev_f1_list), max(test_f1_list)
#     best_dev_beta, best_test_beta = dev_f1_list.index(best_dev_f1), test_f1_list.index(best_test_f1)
#     logger.info(f'The best dev f1:{best_dev_f1}, the best test f1: {best_test_f1}')
#     logger.info(dev_f1_list)
#     logger.info(f'The best dev beta:{beta_list[best_dev_beta]}, the best test beta: {beta_list[best_test_beta]}')
#
#     # 先确定alpha再确定beta（beta=0）
#     dev_f1_list, test_f1_list = [], []
#     args.beta = beta_list[best_dev_beta]
#     for a in alpha_list:
#         args.alpha = a
#         logger.info(f'alpha is {a}, beta is {args.beta}')
#         dev_results, test_results = do_eval(model, dev_dataloader, test_dataloader)
#
#         dev_f1_list.append(dev_results['quadruple']['f1'])
#         test_f1_list.append(test_results['quadruple']['f1'])
#     best_dev_f1, best_test_f1 = max(dev_f1_list), max(test_f1_list)
#     best_dev_alpha, best_test_alpha = dev_f1_list.index(best_dev_f1), test_f1_list.index(best_test_f1)
#     logger.info(f'The best dev f1:{best_dev_f1}, the best test f1: {best_test_f1}')
#     logger.info(dev_f1_list)
#     logger.info(f'The best dev alpha:{alpha_list[best_dev_alpha]}, the best test alpha: {alpha_list[best_test_alpha]}')


def do_inference(reviews):
    # ##########init model##########
    logger.info("Building MRC-CLRI Model...")
    category_list = get_aspect_category(args.task, args.data_type)
    args.category_dim = len(category_list[0])

    # category and sentiment num_list
    res_lists = get_category_sentiment_num_list(args)
    args.category_num_list = res_lists[0]
    args.sentiment_num_list = res_lists[-1]

    # detect device available for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load checkpoint
    model = torch.load(args.checkpoint_path, map_location=device)
    model = model.to(device)

    # print(dir(tokenizer))
    print(f"tokenize分词：{tokenizer.tokenize(reviews[0])}")

    trainer = ACOSTrainer(logger, model, None, None, tokenizer, args)
    trainer.inference(reviews)


if __name__ == '__main__':
    # ##########init args##########
    args = init_args()

    # ##########seed##########
    seed_everything(args.seed)

    # ##########创建目录##########
    output_path = os.path.join(args.output_dir, args.task, args.data_type)
    log_path = os.path.join(args.log_dir, args.task, args.data_type)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # ##########init logger##########
    log_path = os.path.join(log_path, time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()) + ".log")
    logger = get_logger(log_path)

    # print args
    logger.info(args)

    # tokenizer
    # tokenizer = BertTokenizer.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if args.do_train:
        do_train()
    # if args.do_test:
    #     do_test()
    # if args.do_optimized:
    #     do_optimized()
    if args.do_inference:
        text = ['三元酸奶真好喝！']
        do_inference(text)
