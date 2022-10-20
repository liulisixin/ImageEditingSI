import torch
import torch.nn as nn
import functools
from models.networks import get_norm_layer, init_net, UnetReplaceLight


##############################################################################
# Model for three_decoder or two decoder
##############################################################################
def define_net_three_decoder(input_nc, output_nc, ngf, norm='batch', use_dropout_encoder=False,
                             use_dropout_decoder=False, init_type='normal',
                             init_gain=0.02, gpu_ids=[],
                             output_light_condition=True, output_ori_shading=True):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout_encoder (bool) -- if use dropout layers in encoder.
        use_dropout_decoder (bool) -- if use dropout layers in decoder.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
        output_light_condition (bool) -- if output light condition.
        output_ori_shading (bool) -- if output original shading.
    Returns a generator
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    net = UnetDoubleDecoder_three_decoder(input_nc, output_nc, 8, ngf, norm_layer=norm_layer,
                                          use_dropout_encoder=use_dropout_encoder,
                                          use_dropout_decoder=use_dropout_decoder,
                                          output_light_condition=output_light_condition,
                                          output_ori_shading=output_ori_shading)

    return init_net(net, init_type, init_gain, gpu_ids)



class UnetDoubleDecoder_three_decoder(nn.Module):
    """Create a Unet-based generator with multiple decoders"""
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout_encoder=False,
                 use_dropout_decoder=False, output_light_condition=True, output_ori_shading=True):
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
        super(UnetDoubleDecoder_three_decoder, self).__init__()
        # construct unet structure
        unet_block = UnetReplaceLight(ngf * 8, norm_layer=norm_layer,
                                      output_light_condition=output_light_condition)
        skip = UnetDoubleSkipConnectionBlock_three_decoder_no_share_weight
        unet_block = skip(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                          norm_layer=norm_layer, innermost=True,
                          output_light_condition=output_light_condition,
                          output_ori_shading=output_ori_shading)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            # may has dropout
            unet_block = skip(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                              norm_layer=norm_layer, use_dropout_encoder=use_dropout_encoder,
                              use_dropout_decoder=use_dropout_decoder,
                              output_light_condition=output_light_condition,
                              output_ori_shading=output_ori_shading)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = skip(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                          norm_layer=norm_layer,
                          output_light_condition=output_light_condition,
                          output_ori_shading=output_ori_shading)
        unet_block = skip(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                          norm_layer=norm_layer,
                          output_light_condition=output_light_condition,
                          output_ori_shading=output_ori_shading)
        unet_block = skip(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                          norm_layer=norm_layer,
                          output_light_condition=output_light_condition,
                          output_ori_shading=output_ori_shading)
        self.model = skip(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                          outermost=True, norm_layer=norm_layer,
                          output_light_condition=output_light_condition,
                          output_ori_shading=output_ori_shading)  # add the outermost layer

    def forward(self, x_image, x_new_light):
        """Standard forward"""
        return self.model(x_image, x_new_light)


class UnetDoubleSkipConnectionBlock_three_decoder_no_share_weight(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout_encoder=False,
                 use_dropout_decoder=False, output_light_condition=True, output_ori_shading=True):
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
        super(UnetDoubleSkipConnectionBlock_three_decoder_no_share_weight, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.output_light_condition = output_light_condition
        self.output_ori_shading = output_ori_shading
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
        uprelu1 = nn.ReLU(False)
        upnorm1 = norm_layer(outer_nc)
        uprelu2_1 = nn.ReLU(False)
        upnorm2_1 = norm_layer(outer_nc)
        uprelu2_2 = nn.ReLU(False)
        upnorm2_2 = norm_layer(outer_nc)

        if outermost:
            upconv1 = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            upconv2_1 = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                         kernel_size=4, stride=2,
                                         padding=1)
            upconv2_2 = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                         kernel_size=4, stride=2,
                                         padding=1)
            down = [downconv]
            up1 = [uprelu1, upconv1, nn.Sigmoid()]
            up2_1 = [uprelu2_1, upconv2_1, nn.Sigmoid()]
            up2_2 = [uprelu2_2, upconv2_2, nn.Sigmoid()]
        elif innermost:
            upconv1 = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            upconv2_1 = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            upconv2_2 = nn.ConvTranspose2d(inner_nc, outer_nc,
                                         kernel_size=4, stride=2,
                                         padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up1 = [uprelu1, upconv1, upnorm1]
            up2_1 = [uprelu2_1, upconv2_1, upnorm2_1]
            up2_2 = [uprelu2_2, upconv2_2, upnorm2_2]
        else:
            upconv1 = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            upconv2_1 = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            upconv2_2 = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                         kernel_size=4, stride=2,
                                         padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up1 = [uprelu1, upconv1, upnorm1]
            up2_1 = [uprelu2_1, upconv2_1, upnorm2_1]
            up2_2 = [uprelu2_2, upconv2_2, upnorm2_2]

            if use_dropout_encoder:
                down = down + [nn.Dropout(0.5)]
            if use_dropout_decoder:
                up1 = up1 + [nn.Dropout(0.5)]
                up2_1 = up2_1 + [nn.Dropout(0.5)]
                up2_2 = up2_2 + [nn.Dropout(0.5)]

        self.down = nn.Sequential(*down)
        self.submodule = submodule
        self.up1 = nn.Sequential(*up1)
        if self.output_ori_shading:
            self.up2_1 = nn.Sequential(*up2_1)
        self.up2_2 = nn.Sequential(*up2_2)

    def forward(self, x_image, x_new_light):
        """
        1: for reflectance
        2_1: for original shading
        2_2: for new shading
        """
        x_down = self.down(x_image)

        if self.innermost:
            if self.output_light_condition:
                y_ori_light, y_up_with_new_light = self.submodule(x_down, x_new_light)
            else:
                y_up_with_new_light = self.submodule(x_down, x_new_light)
            y_up1 = self.up1(x_down)
            if self.output_ori_shading:
                y_up2_1 = self.up2_1(x_down)
            y_up2_2 = self.up2_2(y_up_with_new_light)
        else:
            if self.output_light_condition:
                if self.output_ori_shading:
                    y_ori_light, y_up1, y_up2_1, y_up2_2 = self.submodule(x_down, x_new_light)
                else:
                    y_ori_light, y_up1, y_up2_2 = self.submodule(x_down, x_new_light)
            else:
                if self.output_ori_shading:
                    y_up1, y_up2_1, y_up2_2 = self.submodule(x_down, x_new_light)
                else:
                    y_up1, y_up2_2 = self.submodule(x_down, x_new_light)
            y_up1 = self.up1(y_up1)
            if self.output_ori_shading:
                y_up2_1 = self.up2_1(y_up2_1)
            y_up2_2 = self.up2_2(y_up2_2)

        if self.outermost:
            if self.output_light_condition:
                if self.output_ori_shading:
                    return y_ori_light, y_up1, y_up2_1, y_up2_2
                else:
                    return y_ori_light, y_up1, y_up2_2
            else:
                if self.output_ori_shading:
                    return y_up1, y_up2_1, y_up2_2
                else:
                    return y_up1, y_up2_2
        else:
            # add skip connections
            if self.output_light_condition:
                if self.output_ori_shading:
                    return y_ori_light, torch.cat([x_image, y_up1], 1), torch.cat([x_image, y_up2_1], 1), \
                           torch.cat([x_image, y_up2_2], 1)
                else:
                    return y_ori_light, torch.cat([x_image, y_up1], 1), \
                           torch.cat([x_image, y_up2_2], 1)
            else:
                if self.output_ori_shading:
                    return torch.cat([x_image, y_up1], 1), torch.cat([x_image, y_up2_1], 1), \
                           torch.cat([x_image, y_up2_2], 1)
                else:
                    return torch.cat([x_image, y_up1], 1), \
                           torch.cat([x_image, y_up2_2], 1)

