import torch
from .base_model import BaseModel
# from torchviz import make_dot
import os
from models.networks_1to1 import define_net_one_decoder
from models.networks import PanTiltLoss, L1_LPIPS_SSIM


class OneDecoderModel(BaseModel):
    """
    This class implements the model with a single decoder.
    """
    def __init__(self, opt):
        """
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['angular', 'color', 'relighted']
        if opt.use_discriminator:
            self.loss_names.extend(['G_GAN', 'D_real', 'D_fake'])
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['Image_input', 'Relighted_gt', 'Relighted_predict']
        if not opt.isTrain:
            self.visual_names.extend(['light_position_color_original', 'light_position_color_new',
                                      'light_position_color_predict'])
            if opt.special_test:
                self.visual_names = ['Image_input', 'Relighted_predict', 'light_position_color_predict',
                                     'light_position_color_new']
        # specify the models you want to save to the disk. The training/test scripts will call
        # <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain and opt.use_discriminator:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        self.netG = define_net_one_decoder(opt.input_nc, opt.output_nc, opt.ngf, opt.norm,
                                           not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # for plotting model
        self.expr_dir = os.path.join(opt.checkpoints_dir, opt.name)

        if self.isTrain:
            # define loss functions
            # self.criterionAngular = networks.AngularLoss()
            if opt.main_loss_function == 'L1':
                main_loss_function = torch.nn.L1Loss()
            elif opt.main_loss_function == 'L2':
                main_loss_function = torch.nn.MSELoss()
            elif opt.main_loss_function == 'L1_LPIPS_SSIM':
                main_loss_function = L1_LPIPS_SSIM(opt.gpu_ids)
            else:
                raise Exception('main_loss_function error')
            self.criterionAngular = PanTiltLoss()
            self.weight_angular = opt.loss_weight_angular
            self.criterionColor = torch.nn.L1Loss()
            self.weight_color = opt.loss_weight_color

            self.criterionRelighted = main_loss_function
            self.weight_relighted = opt.loss_weight_relighted

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    # def plot_model(self):
    #     # generator
    #     x1 = torch.rand(12, 3, 256, 256).to(self.device)
    #     x2 = torch.rand(12, 7).to(self.device)
    #     y = self.netG(x1, x2)
    #     g = make_dot(y, params=dict(self.netG.named_parameters()))
    #     g.render(filename='netG', directory=self.expr_dir, view=False)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        ['Image_input', 'light_position_color_new', 'light_position_color_original', 'Reflectance_output',
        'Shading_output', 'Image_relighted', 'scene_label', 'num_imgs_in_one_batch',
        'Shading_ori']
        """
        self.input = {}
        for key, data in input.items():
            if key != 'scene_label':
                self.input[key] = data.to(self.device)
            else:
                self.input[key] = data

        self.image_paths = input['scene_label']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.predict = {}

        self.predict['light_position_color_predict'], self.predict['Relighted_predict'], \
            = self.netG(self.input['Image_input'], self.input['light_position_color_new'])

        # for visdom
        self.Image_input = self.input['Image_input']
        if self.isTrain or not self.opt.special_test:
            self.Relighted_gt = self.input['Image_relighted']

        self.Relighted_predict = self.predict['Relighted_predict']

        self.light_position_color_original = self.input['light_position_color_original']
        self.light_position_color_predict = self.predict['light_position_color_predict']
        self.light_position_color_new = self.input['light_position_color_new']

    def backward_G(self):
        self.loss_angular = self.weight_angular * self.criterionAngular(self.light_position_color_predict[:, :4],
                                                                       self.light_position_color_original[:, :4])
        self.loss_color = self.weight_color * self.criterionColor(self.light_position_color_predict[:, 4:],
                                                                 self.light_position_color_original[:, 4:])
        self.loss_relighted = self.weight_relighted * self.criterionRelighted(self.Relighted_predict,
                                                                             self.Relighted_gt)
        self.loss_weighted_total = self.loss_angular + self.loss_color + self.loss_relighted
        self.loss_weighted_total.backward()

    def optimize_parameters(self, epoch=0, iter=0):
        self.forward()  # compute fake images: G(A)
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights

    def calculate_val_loss(self):
        self.loss_relit_validation = self.criterionRelighted(self.Relighted_predict,
                                                             self.Relighted_gt)
        return self.loss_relit_validation
