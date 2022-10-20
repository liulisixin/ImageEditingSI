from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def __init__(self):
        super(TrainOptions, self).__init__()
        which_experiment = 'exp_1to1'
        self.which_experiment = which_experiment
        self.isTrain = True

        # Setting for dataset
        self.anno = 'data/check_generate_anno/SID1_train.txt'  # the anno file from prepare_dataset.py
        self.anno_validation = 'data/check_generate_anno/SID1_val_pairs.txt'
        # Setting for GPU
        number_gpus = 1
        self.gpu_ids = [i for i in range(number_gpus)]
        # parameters for batch
        self.batch_size = 96
        self.num_batch_back_prop = 1  # number of batch in each back propagation.

        # Setting for the optimizer
        self.lr_policy = 'step'   #   learning rate policy. [linear | step | plateau | cosine]
        self.lr = 0.0002  # initial learning rate for adam
        self.lr_d = self.lr
        self.lr_decay_ratio = 0.5  # decay ratio in step scheduler.
        self.n_epochs = 250   #100 number of epochs with the initial learning rate
        self.n_epochs_decay = 0   #100 when using 'linear', number of epochs to linearly decay learning rate to zero
        self.lr_decay_iters = 100  # when using 'step', multiply by a gamma every lr_decay_iters iterations
        self.optimizer_type = 'Adam'   # 'Adam', 'SGD'
        self.beta1 = 0.5  # momentum term of adam
        self.adam_eps = 1e-8

        # Setting for continuing the training.
        self.continue_train = False  # continue training: load the latest model
        self.epoch = '100'  # default='latest', which epoch to load? set to latest to use latest cached model
        self.load_iter = 0  # default='0', which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]
        if self.continue_train:
            self.epoch_count = int(self.epoch)
        else:
            self.epoch_count = 1  # the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.model_modify_layer = []
        self.modify_layer = len(self.model_modify_layer) != 0

        # Setting for the dataset
        self.dataset_mode = 'relighting_single_image'  # name of the dataset
        self.name = which_experiment  # name of the experiment. It decides where to store samples and models

        self.model_name = 'one_decoder'  # ['relighting' | 'intrinsic_decomposition']
        self.no_dropout = False  # old option: no dropout for the model
        self.output_light_condition = True
        self.norm = 'batch'  # instance normalization or batch normalization [instance | batch | none]

        # parameters for loss functions
        self.filter_loss_epoch = 0  # filter some loss (reflectance consistency) before this number of epoch.
        self.main_loss_function = 'L1_LPIPS_SSIM'  # choose using L2 or L1 during the training
        # self.ref_consistency_type = "direct"  # "gradient" or "direct"
        if self.output_light_condition:
            self.loss_weight_angular = 1.0
            self.loss_weight_color = 1.0
        self.loss_weight_reflectance = 1.0
        self.loss_weight_shading_new = 1.0
        self.loss_weight_relighted = 1.0
        self.has_ref_consistency = False
        if self.has_ref_consistency:
            self.loss_weight_ref_consistency = 1.0
        self.share_weight_in_shading = False
        # define a relit consistency
        self.has_relit_consistency = False
        self.has_gradient_loss = False

        # data augmentation
        self.preprocess = 'none'   # 'resize_and_crop'   # scaling and cropping of images at load time
        # [resize_and_crop | crop | scale_width | scale_width_and_crop | none]
        self.load_size = 256   # scale images to this size
        self.crop_size = 256   # then crop to this size
        self.no_flip = True   # if specified, do not flip the images for data augmentation

        # dataloader
        self.serial_batches = False   # if true, takes images in order to make batches, otherwise takes them randomly
        self.num_threads = 6   # threads for loading data

        # save model and output images
        self.save_epoch_freq = 25  # frequency of saving checkpoints at the end of epochs
        self.save_latest = False
        self.save_optimizer = True
        self.load_optimizer = True

        # visdom and HTML visualization parameters
        self.display_env = self.name
        self.save_and_show_by_epoch = True
        self.display_freq = 4000  # frequency of showing training results on screen')
        self.display_ncols = 5  # if positive, display all images in a single visdom web panel with certain number of images per row.')
        self.display_id = 1  # window id of the web display')
        self.display_server = "http://localhost"  # visdom server of the web display')

        self.display_port = 8097  # visdom port of the web display')
        self.update_html_freq = 4000  # frequency of saving training results to html')
        self.print_freq = 4000  # frequency of showing training results on console')
        self.no_html = False  # do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')

