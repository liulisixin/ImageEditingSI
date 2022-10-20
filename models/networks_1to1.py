import torch
import torch.nn as nn
import functools
from models.networks import get_norm_layer, init_net, UnetReplaceLight

##############################################################################
# Model for one_decoder
##############################################################################
def define_net_one_decoder(input_nc, output_nc, ngf, norm='batch', use_dropout=False, init_type='normal',
                             init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Returns a generator
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    net = UnetDoubleDecoder_one_decoder(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)

    return init_net(net, init_type, init_gain, gpu_ids)


class UnetDoubleDecoder_one_decoder(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet.
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetDoubleDecoder_one_decoder, self).__init__()
        # construct unet structure
        unet_block = UnetReplaceLight(ngf * 8, norm_layer=norm_layer)
        skip = UnetDoubleSkipConnectionBlock_one_decoder
        unet_block = skip(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                   norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = skip(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                       norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = skip(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                                   norm_layer=norm_layer)
        unet_block = skip(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                                   norm_layer=norm_layer)
        unet_block = skip(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                                   norm_layer=norm_layer)
        self.model = skip(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                                   outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, x_image, x_new_light):
        """Standard forward"""
        return self.model(x_image, x_new_light)


class UnetDoubleSkipConnectionBlock_one_decoder(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetDoubleSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetDoubleSkipConnectionBlock_one_decoder, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        # except the outermost layer, input_nc should be equal to outer_nc
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, False)
        downnorm = norm_layer(inner_nc)
        uprelu2 = nn.ReLU(False)
        upnorm2 = norm_layer(outer_nc)

        if outermost:
            upconv2 = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                         kernel_size=4, stride=2,
                                         padding=1)
            down = [downconv]
            up2_2 = [uprelu2, upconv2, nn.Sigmoid()]
        elif innermost:
            upconv2 = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up2_2 = [uprelu2, upconv2, upnorm2]
        else:
            upconv2 = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up2_2 = [uprelu2, upconv2, upnorm2]

            if use_dropout:
                updrop2 = nn.Dropout(0.5)
                up2_2 = up2_2 + [updrop2]

        self.down = nn.Sequential(*down)
        self.submodule = submodule
        self.up2_2 = nn.Sequential(*up2_2)

    def forward(self, x_image, x_new_light):
        x_down = self.down(x_image)

        if self.innermost:
            y_ori_light, y_up_with_new_light = self.submodule(x_down, x_new_light)
            y_up2_2 = self.up2_2(y_up_with_new_light)
        else:
            y_ori_light, y_up2_2 = self.submodule(x_down, x_new_light)
            y_up2_2 = self.up2_2(y_up2_2)

        if self.outermost:
            return y_ori_light, y_up2_2
        else:
            # add skip connections
            return y_ori_light, torch.cat([x_image, y_up2_2], 1)
