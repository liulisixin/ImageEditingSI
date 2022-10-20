from .base_options import BaseOptions

class TestQuantitiveOptions(BaseOptions):
    def __init__(self):
        super(TestQuantitiveOptions, self).__init__()
        which_experiment = 'exp_three_decoders'   # ['relighting' | 'intrinsic_decomposition']
        self.anno = 'data/check_generate_anno/SID1_test_quantitative_pairs.txt'  # the anno file from prepare_dataset.py
        # save some general setting here.
        if which_experiment == 'exp':
            self.name = 'exp_1to1_vidit'  # name of the experiment. It decides where to store samples and models
            self.model_name = 'one_decoder'  # ['relighting_two_stage' | 'relighting_one_decoder']
            self.no_dropout = False  # old option: no dropout for the model
            self.norm = 'batch'  # instance normalization or batch normalization [instance | batch | none]
            self.use_discriminator = False
            # train parameter
            self.batch_size = 1  # batch size # 6
            self.metric_list = ['Relighted', 'light_position_color', 'Input_and_relighted_gt']
            # select which model to load, set continue_train = True to load the weight
            self.continue_train = True  # continue training: load the latest model
            self.epoch = 'best'  # default='latest', which epoch to load? set to latest to use latest cached model
            self.load_iter = 0  # default='0', which iteration to load? if load_iter > 0, the code will load models by
            # iter_[load_iter]; otherwise, the code will load models by [epoch]
        elif which_experiment == 'exp_two_decoders':
            self.name = 'exp_1to2_vidit'  # name of the experiment. It decides where to store samples and models
            self.model_name = 'three_decoder'  # ['relighting_two_stage' | 'relighting_one_decoder']
            self.use_dropout_encoder = True
            self.use_dropout_decoder = True
            self.output_light_condition = True
            self.output_ori_shading = False
            self.norm = 'batch'  # instance normalization or batch normalization [instance | batch | none]
            # train parameter
            self.batch_size = 1  # batch size # 6
            self.metric_list = ['Relighted', 'light_position_color', 'Reflectance', 'Shading_new',
                                'Input_and_relighted_gt']
            # select which model to load, set continue_train = True to load the weight
            self.continue_train = True  # continue training: load the latest model
            self.epoch = 'best'  # default='latest', which epoch to load? set to latest to use latest cached model
            self.load_iter = 0  # default='0', which iteration to load? if load_iter > 0, the code will load models by
            # iter_[load_iter]; otherwise, the code will load models by [epoch]
        elif which_experiment == 'exp_three_decoders':
            self.name = 'exp_1to3'  # name of the experiment. It decides where to store samples and models
            self.model_name = 'three_decoder'  # ['relighting_two_stage' | 'relighting_one_decoder']
            self.use_dropout_encoder = True
            self.use_dropout_decoder = True
            self.output_light_condition = True
            self.output_ori_shading = True
            self.norm = 'batch'  # instance normalization or batch normalization [instance | batch | none]
            # train parameter
            self.batch_size = 1  # batch size # 6
            self.metric_list = ['Relighted', 'light_position_color', 'Reflectance', 'Shading_ori', 'Shading_new',
                                'Reconstruct', 'Input_and_relighted_gt']
            # select which model to load, set continue_train = True to load the weight
            self.continue_train = True  # continue training: load the latest model
            self.epoch = 'best'  # default='latest', which epoch to load? set to latest to use latest cached model
            self.load_iter = 0  # default='0', which iteration to load? if load_iter > 0, the code will load models by
            # iter_[load_iter]; otherwise, the code will load models by [epoch]
        else:
            raise Exception('which_experiment setting error')

        self.light_type = "pan_tilt_color"  # ["pan_tilt_color" | "Spherical_harmonic"]
        self.model_modify_layer = []
        self.modify_layer = len(self.model_modify_layer) != 0
        self.constrain_intrinsic = True
        self.cycle_ref_cons = False
        self.two_discriminator = False

        self.dataset_mode = 'relighting_single_image_test'  # name of the dataset
        self.results_dir = './results/'  # str, default='./results/', help='saves results here.')
        self.isTrain = False
        self.gpu_ids = [0]

        # self.ntest = 1   # int, default=float("inf"), help='# of test examples.')
        # Dropout and Batchnorm has different behavioir during training and test.
        self.eval = True   # use eval mode during test time.
        self.num_test = float("inf") # 100   # how many test images to run
        # dataloader
        self.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        self.num_threads = 1   # test code only supports num_threads = 1
        # data augmentation
        self.preprocess = 'none'  # 'resize_and_crop'   # scaling and cropping of images at load time
        # [resize_and_crop | crop | scale_width | scale_width_and_crop | none]
        self.crop_size = 256  # then crop to this size
        self.load_size = self.crop_size  # scale images to this size
        self.no_flip = True  # if specified, do not flip the images for data augmentation

        self.display_id = -1  # no visdom display; the test code saves the results to a HTML file.

        self.fix_seed = False  # fix random seed of relighting
        self.special_test = False  # special test for pictures from other datasets.
