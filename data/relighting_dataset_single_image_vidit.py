from data.base_dataset import BaseDataset, get_params, get_transform
from util.util import PARA_NOR
import math
import torch
from util.k_to_rgb import convert_K_to_RGB
from PIL import Image
import os.path


def read_anno(anno_filename):
    """
    Read the anno/*.txt
    """
    # read lines from anno
    f = open(anno_filename, 'r')
    file_names = []
    for line in f.readlines():
        line = line.strip('\n')
        file_names.append(line)
    return file_names


def transform_light(pan, tilt, color_temp):
    # transform light position to cos and sin
    light_position = [math.cos(pan), math.sin(pan), math.cos(tilt), math.sin(tilt)]
    # normalize the light position to [0, 1]
    light_position[:2] = [x * PARA_NOR['pan_a'] + PARA_NOR['pan_b'] for x in light_position[:2]]
    light_position[2:] = [x * PARA_NOR['tilt_a'] + PARA_NOR['tilt_b'] for x in light_position[2:]]
    # transform light temperature to RGB, and normalize it.
    light_color = list(map(lambda x: x / 255.0, convert_K_to_RGB(color_temp)))
    light_position_color = light_position + light_color
    return torch.tensor(light_position_color)


class RelightingDatasetSingleImageVidit(BaseDataset):
    """A dataset class for relighting dataset.
       This dataset read data image by image.
    """

    def __init__(self, opt, validation=False):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        if validation:
            anno_file = opt.anno_validation
        else:
            anno_file = opt.anno

        # for vidit dataset
        self.dataroot = self.opt.dataroot_vidit
        # North, 6500K   --->  East, 4500K
        self.input_pan = 180
        self.output_pan = 90
        self.tilt = 45.0
        self.input_color = 6500
        self.output_color = 4500
        self.input_light = transform_light(self.input_pan, self.tilt, self.input_color)
        self.output_light = transform_light(self.output_pan, self.tilt, self.output_color)

        self.file_names = read_anno(anno_file)

        # get the parameters of data augmentation
        img_size = self.opt.img_size
        transform_params = get_params(self.opt, img_size)
        self.img_transform = get_transform(self.opt, transform_params)

    def get_data(self, file_name):
        data = {}  # output dictionary

        data['scene_label'] = file_name
        data['light_position_color_original'] = self.input_light
        data['light_position_color_new'] = self.output_light

        def read_image(dataroot, component, file_name, img_transform):
            component_path = "{}{}/{}".format(dataroot, component, file_name)
            if not os.path.exists(component_path):
                raise Exception("RelightingDataset __getitem__ error")

            img_component = Image.open(component_path).convert('RGB')
            img_tensor = img_transform(img_component)
            return img_tensor

        data['Image_input'] = read_image(self.dataroot, 'input', file_name, self.img_transform)
        data['Image_relighted'] = read_image(self.dataroot, 'target', file_name, self.img_transform)
        return data

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains
            'Image_input': ,
            'light_position_color_new': ,
            'light_position_color_original': ,
            'Reflectance_output': ,
            'Shading_output': ,
            'Shading_ori': ,
            'Image_relighted': ,
            'scene_label': ,
        """
        # get parameters
        index_file_names = index
        # get one image
        file_name = self.file_names[index_file_names]
        data = self.get_data(file_name)

        return data

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.file_names)

