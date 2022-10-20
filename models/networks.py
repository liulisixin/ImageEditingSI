import torch
import torch.nn as nn
import functools
from torch.nn import init
from torch.optim import lr_scheduler
import torch.nn.functional as F
import lpips
from pytorch_msssim import SSIM
from util.util import PARA_NOR

###############################################################################
# Functions
###############################################################################


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=opt.lr_decay_ratio)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


class UnetReplaceLight(nn.Module):
    """Defines the center submodule of Unet to replace the light condition.
            X -------------------identity----------------------
            |-- downsampling -- original light condition                  |
            |                        new light condition  -- upsampling --|
    """
    def __init__(self, nc_two_head, norm_layer=nn.BatchNorm2d, output_light_condition=True):
        super(UnetReplaceLight, self).__init__()
        self.output_light_condition = output_light_condition
        input_nc = nc_two_head
        output_nc = nc_two_head
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if self.output_light_condition:
            conv_down = [nn.LeakyReLU(0.2, False), nn.Conv2d(input_nc, input_nc // 2, kernel_size=3,
                                 stride=1, padding=1, bias=use_bias), nn.LeakyReLU(0.2, False)]
            fc_down = [nn.Linear(256, 7)]
            self.conv_down = nn.Sequential(*conv_down)
            self.fc_down = nn.Sequential(*fc_down)

        fc_up = [nn.Linear(7, 256)]
        conv_up = [nn.ReLU(False), nn.ConvTranspose2d(input_nc // 2, input_nc,
                                                     kernel_size=3, stride=1, padding=1), norm_layer(output_nc)]
        # conv_up = [nn.ReLU(True), nn.ConvTranspose2d(input_nc // 2, input_nc,
        #                                              kernel_size=3, stride=1, padding=1), norm_layer(output_nc)]
        self.fc_up = nn.Sequential(*fc_up)
        self.conv_up = nn.Sequential(*conv_up)

    def forward(self, x_down, x_new_light):
        if self.output_light_condition:
            y_ori_light = self.conv_down(x_down)
            y_ori_light = y_ori_light.view(y_ori_light.size(0), -1)
            y_ori_light = self.fc_down(y_ori_light)
            # need to be sigmoid.
            y_ori_light = torch.sigmoid(y_ori_light)

        y_up = self.fc_up(x_new_light)
        y_up = y_up.unsqueeze(2)
        y_up = y_up.unsqueeze(3)
        y_up = self.conv_up(y_up)

        if self.output_light_condition:
            return y_ori_light, y_up
        else:
            return y_up


##############################################################################
# Losses
##############################################################################

class PanTiltLoss(nn.Module):
    # Pan and tilt should use the angular loss because pan can be any values when tilt = 0
    # The normalized angle is [cos(pan)/2+0.5, sin(pan)/2+0.5, cos(tilt), sin(tilt)]
    def __init__(self):
        super(PanTiltLoss, self).__init__()

    def pan_tilt_to_vector(self, tri_nor):
        # tri_nor: [cos(pan) / 2 + 0.5, sin(pan) / 2 + 0.5, cos(tilt), sin(tilt)]
        tri = torch.zeros(tri_nor.size()).cuda()
        tri[:, :2] = (tri_nor[:, :2] - PARA_NOR['pan_b']) / PARA_NOR['pan_a']
        tri[:, 2:] = (tri_nor[:, 2:] - PARA_NOR['tilt_b']) / PARA_NOR['tilt_a']
        # tri: [cos(pan), sin(pan), cos(tilt), sin(tilt)]
        vector = torch.zeros(tri.size()[0], 3).cuda()
        vector[:, 0] = torch.mul(tri[:, 3], tri[:, 0])
        vector[:, 1] = torch.mul(tri[:, 3], tri[:, 1])
        vector[:, 2] = tri[:, 2]
        return vector

    def forward(self, pan_tilt_pred, pan_tilt_target):
        vector_pred = self.pan_tilt_to_vector(pan_tilt_pred)
        vector_target = self.pan_tilt_to_vector(pan_tilt_target)

        distance = torch.sqrt(F.mse_loss(vector_pred, vector_target))
        return distance


class L1_LPIPS_SSIM(nn.Module):
    def __init__(self, gpu_ids):
        super(L1_LPIPS_SSIM, self).__init__()
        net = lpips.LPIPS(net='alex')
        if len(gpu_ids) > 0:
            assert (torch.cuda.is_available())
            net.to(gpu_ids[0])
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
        self.loss_lpips = net
        self.loss_l1 = torch.nn.L1Loss()
        self.loss_ssim = SSIM(data_range=1.0, size_average=True, channel=3)

    def forward(self, predict, target):
        distance_lpips = self.loss_lpips.forward(predict * 2 - 1, target * 2 - 1)  # change [0,+1] to [-1,+1]
        distance_lpips = distance_lpips.mean()
        distance_l1 = self.loss_l1(predict, target)
        distance_ssim = 1.0 - self.loss_ssim(predict, target)
        distance_total = distance_lpips + distance_l1 + distance_ssim
        return distance_total


