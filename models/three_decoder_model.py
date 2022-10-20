import torch
from .base_model import BaseModel
from models.networks_1to3 import define_net_three_decoder
from torchviz import make_dot
import os
from models.networks import PanTiltLoss, L1_LPIPS_SSIM


class ThreeDecoderModel(BaseModel):
    """
    This class implements the model with multiple decoders.
    """
    def __init__(self, opt):
        """
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.output_light_condition = self.opt.output_light_condition
        self.output_ori_shading = self.opt.output_ori_shading
        self.gt_has_intrinsic = self.opt.gt_has_intrinsic
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        if self.isTrain:
            self.loss_names = ['relighted']
            if self.opt.cycle_ref_cons:
                self.loss_names.extend(['cycle_ref_cons'])
            if self.gt_has_intrinsic:
                self.loss_names.extend(['reflectance', 'shading_new'])
            if self.output_light_condition:
                self.loss_names.extend(['angular', 'color'])
            if self.output_ori_shading:
                self.loss_names.extend(['reconstruct'])
                if self.gt_has_intrinsic:
                    self.loss_names.extend(['shading_ori'])

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['Image_input', 'Relighted_gt',
                             'Reflectance_predict', 'Shading_new_predict', 'Relighted_predict']
        if self.gt_has_intrinsic:
            self.visual_names.extend(['Reflectance_gt', 'Shading_new_gt'])
        if opt.cycle_ref_cons:
            self.visual_names.extend(['Reflectance_cycle_predict'])
        if self.output_ori_shading:
            self.visual_names.extend(['Reconstruct', 'Shading_ori_predict', ])
            if self.gt_has_intrinsic:
                self.visual_names.extend(['Shading_ori_gt'])
        if not opt.isTrain:
            self.visual_names.extend(['light_position_color_original', 'light_position_color_new'])
            if self.output_light_condition:
                self.visual_names.extend(['light_position_color_predict'])
            if opt.special_test:
                self.visual_names = ['Image_input', 'Reflectance_predict', 'Shading_new_predict', 'Relighted_predict', ]
                self.visual_names.extend(['light_position_color_new'])
                if self.output_light_condition:
                    self.visual_names.extend(['light_position_color_predict'])
                if self.output_ori_shading:
                    self.visual_names.extend(['Reconstruct', 'Shading_ori_predict'])
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>

        self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = define_net_three_decoder(opt.input_nc, opt.output_nc, opt.ngf, opt.norm,
                                             opt.use_dropout_encoder, opt.use_dropout_decoder,
                                             opt.init_type, opt.init_gain, self.gpu_ids,
                                             opt.output_light_condition, opt.output_ori_shading)

        self.expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if self.isTrain:
            if opt.main_loss_function == 'L1':
                main_loss_function = torch.nn.L1Loss()
            elif opt.main_loss_function == 'L2':
                main_loss_function = torch.nn.MSELoss()
            elif opt.main_loss_function == 'L1_LPIPS_SSIM':
                main_loss_function = L1_LPIPS_SSIM(opt.gpu_ids)
            else:
                raise Exception('main_loss_function error')
            if self.output_light_condition:
                self.criterionAngular = PanTiltLoss()
                self.weight_angular = opt.loss_weight_angular
                self.criterionColor = torch.nn.L1Loss()
                self.weight_color = opt.loss_weight_color
            if self.output_ori_shading:
                self.criterionShading_ori = main_loss_function
                self.weight_shading_ori = opt.loss_weight_shading_ori
                self.criterionReconstruct = main_loss_function
                self.weight_reconstruct = opt.loss_weight_reconstruct
            self.criterionReflectance = main_loss_function
            self.weight_reflectance = opt.loss_weight_reflectance
            self.criterionShading_new = main_loss_function
            self.weight_shading_new = opt.loss_weight_shading_new
            self.criterionRelighted = main_loss_function
            self.weight_relighted = opt.loss_weight_relighted
            if opt.cycle_ref_cons:
                self.criterionCycleRefCons = main_loss_function
                self.weight_cycle_ref_cons = opt.loss_weight_reflectance
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)


    def plot_model(self):
        x1 = torch.rand(12, 3, 256, 256).to(self.device)
        x2 = torch.rand(12, 7).to(self.device)
        y = self.netG(x1, x2)
        g = make_dot(y, params=dict(self.netG.named_parameters()))
        g.render(filename='espnet_model', directory=self.expr_dir, view=False)

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

        if self.output_light_condition:
            if self.output_ori_shading:
                self.predict['light_position_color_predict'], self.predict['Reflectance_predict'], \
                self.predict['Shading_ori_predict'], self.predict['Shading_new_predict'] \
                    = self.netG(self.input['Image_input'], self.input['light_position_color_new'])
            else:
                self.predict['light_position_color_predict'], self.predict['Reflectance_predict'], \
                self.predict['Shading_new_predict'] \
                    = self.netG(self.input['Image_input'], self.input['light_position_color_new'])
        else:
            if self.output_ori_shading:
                self.predict['Reflectance_predict'], \
                self.predict['Shading_ori_predict'], self.predict['Shading_new_predict'] \
                    = self.netG(self.input['Image_input'], self.input['light_position_color_new'])
            else:
                self.predict['Reflectance_predict'], \
                self.predict['Shading_new_predict'] \
                    = self.netG(self.input['Image_input'], self.input['light_position_color_new'])
        # in order to calculate the relighted image, we need to make sure the range of input image.
        if self.output_ori_shading:
            self.predict['Reconstruct'] = torch.mul(self.predict['Reflectance_predict'],
                                                    self.predict['Shading_ori_predict'])
        self.predict['Relighted_predict'] = torch.mul(self.predict['Reflectance_predict'],
                                                      self.predict['Shading_new_predict'])

        if self.opt.cycle_ref_cons:
            if self.output_light_condition:
                if self.output_ori_shading:
                    _, self.predict['Reflectance_cycle_predict'], _, _ = \
                        self.netG(self.predict['Relighted_predict'], self.input['light_position_color_new'])
                else:
                    _, self.predict['Reflectance_cycle_predict'], _ = \
                        self.netG(self.predict['Relighted_predict'], self.input['light_position_color_new'])
            else:
                if self.output_ori_shading:
                    self.predict['Reflectance_cycle_predict'], _, _ = \
                        self.netG(self.predict['Relighted_predict'], self.input['light_position_color_new'])
                else:
                    self.predict['Reflectance_cycle_predict'], _ = \
                        self.netG(self.predict['Relighted_predict'], self.input['light_position_color_new'])

        # for visdom and output
        self.Image_input = self.input['Image_input']
        if self.opt.isTrain or not self.opt.special_test:
            if self.gt_has_intrinsic:
                self.Reflectance_gt = self.input['Reflectance_output']
                if self.output_ori_shading:
                    self.Shading_ori_gt = self.input['Shading_ori']
                self.Shading_new_gt = self.input['Shading_output']
            self.Relighted_gt = self.input['Image_relighted']
        if self.output_ori_shading:
            self.Reconstruct = self.predict['Reconstruct']
            self.Shading_ori_predict = self.predict['Shading_ori_predict']
        self.Reflectance_predict = self.predict['Reflectance_predict']
        self.Shading_new_predict = self.predict['Shading_new_predict']
        self.Relighted_predict = self.predict['Relighted_predict']
        if self.opt.cycle_ref_cons:
            self.Reflectance_cycle_predict = self.predict['Reflectance_cycle_predict']

        # for light condition
        self.light_position_color_original = self.input['light_position_color_original']
        if self.output_light_condition:
            self.light_position_color_predict = self.predict['light_position_color_predict']
        self.light_position_color_new = self.input['light_position_color_new']

    def backward(self):
        if self.output_light_condition:
            self.loss_angular = self.criterionAngular(self.light_position_color_predict[:, :4],
                                                      self.light_position_color_original[:, :4])
            self.loss_color = self.criterionColor(self.light_position_color_predict[:, 4:],
                                                  self.light_position_color_original[:, 4:])
        if self.gt_has_intrinsic:
            self.loss_reflectance = self.criterionReflectance(self.Reflectance_predict,
                                                              self.Reflectance_gt)
            self.loss_shading_new = self.criterionShading_new(self.Shading_new_predict,
                                                              self.Shading_new_gt)
        if self.output_ori_shading:
            if self.gt_has_intrinsic:
                self.loss_shading_ori = self.criterionShading_ori(self.Shading_ori_predict,
                                                                  self.Shading_ori_gt)
            self.loss_reconstruct = self.criterionReconstruct(self.Reconstruct,
                                                              self.Image_input)
        self.loss_relighted = self.criterionRelighted(self.Relighted_predict,
                                                      self.Relighted_gt)

        if self.opt.cycle_ref_cons:
            self.loss_cycle_ref_cons = self.criterionCycleRefCons(self.Reflectance_predict,
                                                                  self.Reflectance_cycle_predict)

        # combine loss and calculate gradients
        self.loss_weighted_total = torch.tensor(0).to(self.device)
        for name in self.loss_names:
            if isinstance(name, str):
                self.loss_weighted_total = self.loss_weighted_total \
                                           + getattr(self, 'weight_' + name) * getattr(self, 'loss_' + name)

        self.loss_weighted_total.backward()

    def optimize_parameters(self, epoch=0, iter=0):
        self.forward()  # compute fake images: G(A)
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights

    def calculate_val_loss(self):
        self.loss_relit_validation = self.criterionRelighted(self.Relighted_predict,
                                                      self.Relighted_gt)
        return self.loss_relit_validation
