import copy
import os

import torch
from torch.nn import functional as F
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
        self.loss_names = ['l1']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['img1', 'img2', 'fake_flow', 'real_flow']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        self.model_names = ['F']

        self.netF = FlowNet(input_nc=6, ngfs=[64])
        self.netF = init_net(self.netF, opt.gpu_ids)

        # define loss functions
        self.criterionL1 = torch.nn.L1Loss()

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
        self.real_flow = input['flow'].to(self.device)
        self.image_paths = input['img_paths']

    def forward(self):
        self.fake_flow = self.netF(torch.cat((self.img1, self.img2), 1), 0).permute(0, 2, 3, 1)

    def backward(self):
        lambda_l1 = self.opt.lambda_L1

        self.loss_l1 = self.criterionL1(self.fake_flow, self.real_flow) * lambda_l1
        self.loss = self.loss_l1
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
            d = torch.linspace(-1, 1, self.opt.load_size // 4)
            grid_y, grid_x = torch.meshgrid(d, d)
            grid = torch.stack((grid_x, grid_y), 2).unsqueeze(0).to(self.device)

            for i, data_i in enumerate(tqdm(self.eval_dataloader, desc='Eval       ', position=2, leave=False)):
                self.set_input(data_i)
                real_im = self.modelG.netG.model(self.img2)

                fake_flow = self.netF(torch.cat((self.img1, self.img2), 1), 0).permute(0, 2, 3, 1)
                fake_flow_norm = fake_flow / ((self.opt.load_size // 4 - 1.0) / 2.0)
                fake_im = self.modelG.netG.model[14:](
                    F.grid_sample(self.modelG.netG.model[:14](self.img1), grid + fake_flow_norm, padding_mode='border')
                )

                for j in range(len(self.image_paths)):
                    name = self.image_paths[j]
                    input_im = util.tensor2im(self.img1)
                    real_im = util.tensor2im(real_im)
                    fake_im = util.tensor2im(fake_im)
                    util.save_image(input_im, os.path.join(save_dir, 'input', '%s.png' % name), create_dir=True)
                    util.save_image(real_im, os.path.join(save_dir, 'real', '%s.png' % name), create_dir=True)
                    util.save_image(fake_im, os.path.join(save_dir, 'fake', '%s.png' % name), create_dir=True)

        self.netF.train()
