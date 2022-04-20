import copy
import os
import pickle

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm

from data import CustomDatasetDataLoader
from models import networks
from models.base_model import BaseModel
from models.modules.flow_modules import FlowNet
from utils import util


def init_net(net, gpu_ids):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs

    return net


def create_eval_dataloader(opt):
    opt = copy.deepcopy(opt)
    opt.isTrain = False
    opt.serial_batches = True
    opt.batch_size = opt.eval_batch_size
    opt.phase = 'val'
    dataloader = CustomDatasetDataLoader(opt)
    dataloader = dataloader.load_data()
    return dataloader


class FlowModel(BaseModel):
    def __init__(self, opt):
        assert opt.isTrain
        super(FlowModel, self).__init__(opt)

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['l1', 'l1_out']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['img1', 'img2', 'fake_diff', 'real_diff']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        self.model_names = ['F']

        self.netF = FlowNet(input_nc=70, ngfs=[64])
        self.netF = init_net(self.netF, opt.gpu_ids)

        # define loss functions
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionL1Out = torch.nn.L1Loss()

        # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        self.optimizer = torch.optim.Adam(self.netF.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        self.optimizers = []
        self.optimizers.append(self.optimizer)

        self.eval_dataloader = create_eval_dataloader(self.opt)

    def setup_with_G(self, opt, modelG, verbose=True):
        self.modelG = modelG
        self.load_networks(verbose=verbose)
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if verbose:
            self.print_networks()

    def set_input(self, input):
        self.img1 = input['img1'].to(self.device)
        self.img2 = input['img2'].to(self.device)
        self.image_paths = input['img_paths']

    def forward(self):
        b = self.img1.size(0)
        activations = self.modelG.netG.model[:self.opt.layer_idx](torch.cat((self.img1, self.img2), 0)).detach()
        self.real_diff = activations[b:] - activations[:b]
        img1_resized = F.interpolate(self.img1, size=128, mode='bicubic')
        img2_resized = F.interpolate(self.img2, size=128, mode='bicubic')
        self.fake_diff = self.netF(torch.cat((img1_resized, img2_resized, activations[:b]), 1), 0)
        self.real_im = self.modelG.netG.model[self.opt.layer_idx:](activations[b:]).detach()
        self.fake_im = self.modelG.netG.model[self.opt.layer_idx:](activations[:b] + self.fake_diff)

    def backward(self):
        lambda_l1 = self.opt.lambda_L1
        lambda_l1_out = self.opt.lambda_L1_out

        self.loss_l1 = self.criterionL1(self.fake_diff, self.real_diff)
        self.loss_l1_out = self.criterionL1Out(self.fake_im, self.real_im)
        self.loss = self.loss_l1 * lambda_l1 + self.loss_l1_out * lambda_l1_out
        self.loss.backward()

    def optimize_parameters(self, steps):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def evaluate_model(self, step):
        save_dir = os.path.join(self.opt.log_dir, 'eval', str(step))
        os.makedirs(save_dir, exist_ok=True)
        self.netF.eval()

        with torch.no_grad():
            for i, data_i in enumerate(tqdm(self.eval_dataloader, desc='Eval       ', position=2, leave=False)):
                self.set_input(data_i)
                activations = self.modelG.netG.model[:self.opt.layer_idx](self.img1)
                img1_resized = F.interpolate(self.img1, size=128, mode='bicubic')
                img2_resized = F.interpolate(self.img2, size=128, mode='bicubic')
                fake_diff = self.netF(torch.cat((img1_resized, img2_resized, activations), 1), 0)
                real_im = self.modelG.netG.model(self.img2)
                fake_im = self.modelG.netG.model[self.opt.layer_idx:](activations + fake_diff)

                for j in range(len(self.image_paths)):
                    name = self.image_paths[j]
                    input1_im = util.tensor2im(self.img1)
                    input2_im = util.tensor2im(self.img2)
                    real_im = util.tensor2im(real_im)
                    fake_im = util.tensor2im(fake_im)
                    util.save_image(input1_im, os.path.join(save_dir, 'input1', '%s.png' % name), create_dir=True)
                    util.save_image(input2_im, os.path.join(save_dir, 'input2', '%s.png' % name), create_dir=True)
                    util.save_image(real_im, os.path.join(save_dir, 'real', '%s.png' % name), create_dir=True)
                    util.save_image(fake_im, os.path.join(save_dir, 'fake', '%s.png' % name), create_dir=True)

        self.netF.train()
