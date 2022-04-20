import argparse
import pickle
import random
import sys
import time

import numpy as np
import torch
from torch.backends import cudnn
from tqdm import tqdm, trange

from data import create_dataloader
from models import create_model
from utils.logger import Logger


def get_opt():
    parser = argparse.ArgumentParser()
    # basic parameters
    parser.add_argument('--G_opt_path', type=str, required=True)
    parser.add_argument('--dataroot', required=True, help='path to images')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--seed', type=int, default=233, help='random seed')

    # log parameters
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='training logs are saved here')
    parser.add_argument('--tensorboard_dir', type=str, default=None,
                        help='tensorboard is saved here')
    parser.add_argument('--print_freq', type=int, default=100,
                        help='frequency of showing training results on console')
    parser.add_argument('--save_epoch_freq', type=int, default=5,
                        help='frequency of saving checkpoints at the end of epoch')
    parser.add_argument('--epoch_base', type=int, default=1,
                        help='the epoch base of the training (used for resuming)')
    parser.add_argument('--iter_base', type=int, default=0,
                        help='the iteration base of the training (used for resuming)')

    # model parameters
    parser.add_argument('--model', type=str, default='flow', help='choose which model to use')
    parser.add_argument('--layer_idx', type=int, default=14)

    parser.add_argument('--restore_F_path', type=str, default=None,
                        help='the path to restore the generator F')
    parser.add_argument('--restore_O_path', type=str, default=None,
                        help='the path to restore the optimizer')

    # dataset parameters
    parser.add_argument('--dataset_mode', type=str, default='video_forensics',
                        help='chooses how datasets are loaded.')
    parser.add_argument('--serial_batches', action='store_true',
                        help='if true, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
    parser.add_argument('--max_interval', type=int, default=10)
    parser.add_argument('--crop_prob', type=int, default=1, help='crop 적용될 확률')
    parser.add_argument('--crop_scale', type=float, default=0.8, help='살아남는 이미지 최소 비율. 0.7이면 crop 후 0.7~1 이미지가 살아남음')
    parser.add_argument('--shake_scale', type=float, default=0.0, help='이미지 흔들림 정도. 0.5면 crop된 이전 프레임 이미지의 절반이 이동함')
    parser.add_argument('--load_size', type=int, default=512, help='scale images to this size')
    parser.add_argument('--preprocess', type=str, default='resize_and_crop',
                        help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
    parser.add_argument('--crop_separate', action='store_true')
    parser.add_argument('--crop_more', action='store_true')
    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

    # training parameters
    parser.add_argument('--nepochs', type=int, default=5,
                        help='number of epochs with the initial learning rate')  # TODO: hyper-parameter tuning
    parser.add_argument('--nepochs_decay', type=int, default=15,
                        help='number of epochs to linearly decay learning rate to zero')  # TODO: hyper-parameter tuning
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')  # TODO: hyper-parameter tuning
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')  # TODO: hyper-parameter tuning
    parser.add_argument('--lr_policy', type=str, default='linear',
                        help='learning rate policy. [linear | step | plateau | cosine]')
    parser.add_argument('--scheduler_counter', type=str, default='epoch', choices=['epoch', 'iter'],
                        help='which counter to use for the scheduler')

    parser.add_argument('--lambda_L1', type=float, default=1.0)  # TODO: hyper-parameter tuning
    parser.add_argument('--lambda_L1_out', type=float, default=1.0)  # TODO: hyper-parameter tuning

    # evaluation parameters
    parser.add_argument('--eval_batch_size', type=int, default=1, help='the evaluation batch size')

    opt = parser.parse_args()
    opt.isTrain = True
    opt.tensorboard_dir = opt.log_dir if opt.tensorboard_dir is None else opt.tensorboard_dir

    # set gpu ids
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    opt.gpu_ids = sorted(opt.gpu_ids)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])

    return opt


def set_seed(seed):
    cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(opt, dataloader, logger, model):
    start_epoch = opt.epoch_base
    end_epoch = opt.nepochs + opt.nepochs_decay
    total_iter = opt.iter_base

    epoch_tqdm = trange(start_epoch, end_epoch + 1, desc='Epoch      ', position=0, leave=False)
    logger.set_progress_bar(epoch_tqdm)
    for epoch in epoch_tqdm:
        epoch_start_time = time.time()  # timer for entire epoch
        for i, data_i in enumerate(tqdm(dataloader, desc='Batch      ', position=1, leave=False)):
            iter_start_time = time.time()
            total_iter += 1
            model.set_input(data_i)
            model.optimize_parameters(total_iter)

            if total_iter % opt.print_freq == 0:
                losses = model.get_current_losses()
                logger.print_current_errors(epoch, total_iter, losses, time.time() - iter_start_time)
                logger.plot(losses, total_iter)
            if opt.scheduler_counter == 'iter':
                model.update_learning_rate(epoch, total_iter, logger=logger)

        logger.print_info(
            'End of epoch %d / %d \t Time Taken: %.2f sec' % (epoch, end_epoch, time.time() - epoch_start_time))
        if epoch % opt.save_epoch_freq == 0 or epoch == end_epoch:
            model.evaluate_model(total_iter)
            logger.print_info('Saving the model at the end of epoch %d, iters %d' % (epoch, total_iter))
            model.save_networks('latest')
            model.save_networks(epoch)
        if opt.scheduler_counter == 'epoch':
            model.update_learning_rate(epoch, total_iter, logger=logger)


def main():
    opt = get_opt()
    print(' '.join(sys.argv))

    set_seed(opt.seed)
    with open(opt.G_opt_path, 'rb') as f:
        G_opt = pickle.load(f)

    dataloader = create_dataloader(opt)
    G = create_model(G_opt, verbose=False)
    G.setup(G_opt, verbose=False)
    model = create_model(opt)
    model.setup_with_G(opt, G, verbose=False)
    logger = Logger(opt)

    train(opt, dataloader, logger, model)


if __name__ == '__main__':
    main()
