import os

import torch
import torch.nn as nn
import pygan.core.checkpoint as checkpoint
import pygan.core.distributed as dist
import pygan.core.optimizer as optim
from pygan.core.config import cfg
from pygan.core.net import unwrap_model
from pygan.module_dcgan.utils import ImagePool
from pygan.module_dcgan import networks


class BaseModel(object):

    def __init__(self):
        self.is_train = False
        self.loss_names = []
        self.model_names = []
        self.visual_names = []

    def cuda(self, device):
        for name in self.model_names:
            getattr(self, name).cuda(device=device)
        return self

    def parallel(self, device):
        for name in self.model_names:
            model = getattr(self, name)
            model = torch.nn.parallel.DistributedDataParallel(
                module=model, device_ids=[device], output_device=device)
            setattr(self, name, model)
        return self

    def eval(self):
        for name in self.model_names:
            getattr(self, name).eval()

    def train(self):
        for name in self.model_names:
            getattr(self, name).train()

    def get_current_losses(self):
        return [getattr(self, name) for name in self.loss_names], self.loss_names

    def get_current_visuals(self):
        return [getattr(self, name) for name in self.visual_names], self.visual_names

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    @staticmethod
    def complexity(cx):
        return cx

    def __repr__(self):
        return "{}(is_train={})".format(self.__class__.__name__, self.is_train)


class Img2Img(BaseModel):

    def __init__(self, is_train=False):
        self.is_train = is_train

        if self.is_train:
            self.fake_pool = ImagePool(50)
            self.model_names = ["netG", "netD"]
            self.netG = networks.ResnetGenerator()
            self.netD = networks.NLayerDiscriminator()
            self.criterionGAN = networks.GANLoss("lsgan").cuda()
            self.criterionIdt = torch.nn.L1Loss().cuda()
            self.optimizer_G = torch.optim.SGD(
                self.netG.parameters(),
                lr=cfg.OPTIM.BASE_LR,
                momentum=cfg.OPTIM.MOMENTUM,
                weight_decay=cfg.OPTIM.WEIGHT_DECAY,
                dampening=cfg.OPTIM.DAMPENING,
                nesterov=cfg.OPTIM.NESTEROV,
            )
            self.optimizer_D = torch.optim.SGD(
                self.netD.parameters(),
                lr=cfg.OPTIM.BASE_LR,
                momentum=cfg.OPTIM.MOMENTUM,
                weight_decay=cfg.OPTIM.WEIGHT_DECAY,
                dampening=cfg.OPTIM.DAMPENING,
                nesterov=cfg.OPTIM.NESTEROV,
            )
            self.loss_names = ["loss_G_idt", "loss_G_gan", "loss_G", "loss_D_real", "loss_D_fake", "loss_D"]
        else:
            self.model_names = ["netG"]
            self.netG = networks.ResnetGenerator()

        self.visual_names = ["real", "fake"]

    def set_input(self, data):
        self.real = data[0]
        self.peak = data[1]

    def forward(self):
        self.fake = self.netG(self.real)

    def optimize_parameters(self):
        # forward
        self.forward()

        # 1) Update G
        self.set_requires_grad([self.netD], False)
        self.optimizer_G.zero_grad()
        self.loss_G_idt = self.criterionIdt(self.fake, self.peak)
        self.loss_G_gan = self.criterionGAN(self.netD(self.fake), True)
        self.loss_G = (self.loss_G_idt + self.loss_G_gan) * 0.5
        self.loss_G.backward()
        self.optimizer_G.step()

        # 2) Update D
        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()
        fake = self.fake_pool.query(self.fake)
        self.loss_D_real = self.criterionGAN(self.netD(self.real), True)
        self.loss_D_fake = self.criterionGAN(self.netD(fake.detach()), False)
        self.loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5
        self.loss_D.backward()
        self.optimizer_D.step()

    def update_lr(self, lr_G, lr_D):
        optim.set_lr(self.optimizer_G, lr_G)
        optim.set_lr(self.optimizer_D, lr_D)

    def save_networks(self, epoch):
        if not dist.is_master_proc():
            return
        os.makedirs(checkpoint.get_checkpoint_dir(), exist_ok=True)
        state_dict = {
            "epoch": epoch,
            "netG": unwrap_model(self.netG).state_dict(),
            "optimizer_G": self.optimizer_G.state_dict(),
            "netD": unwrap_model(self.netD).state_dict(),
            "optimizer_D": self.optimizer_D.state_dict(),
            "cfg": cfg.dump(),
        }
        state_file = checkpoint.get_checkpoint(epoch + 1)
        torch.save(state_dict, state_file)
        return state_file

    def load_networks(self, state_file):
        err_str = "Checkpoint '{}' not found"
        assert os.path.exists(state_file), err_str.format(state_file)
        state_dict = torch.load(state_file, map_location="cpu")
        unwrap_model(self.netG).load_state_dict(state_dict["netG"])
        if self.is_train:
            self.optimizer_G.load_state_dict(state_dict["optimizer_G"])
            unwrap_model(self.netD).load_state_dict(state_dict["netD"])
            self.optimizer_D.load_state_dict(state_dict["optimizer_D"])
        return state_dict["epoch"]
