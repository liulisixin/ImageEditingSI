from data.base_dataset import BaseDataset, get_params, get_transform
from data.relighting_dataset_single_image import read_anno_single_image, get_data


class RelightingDatasetSingleImageTest(BaseDataset):
    """A dataset class for relighting dataset.
       This dataset read data image by image (for test).
       Read test pairs from anno files. Each scene only has 2 images, one for input and one for relighting.
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
        self.file_names, self.scene_index, self.scenes_list = read_anno_single_image(anno_file)

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
        dataroot = self.dataroot
        img_size = self.opt.img_size
        multiple_replace_image = self.opt.multiple_replace_image
        # get one scene
        index_scene = index
        file_name_input = self.scenes_list[index_scene][0]
        file_name_output = self.scenes_list[index_scene][1]

        # get the parameters of data augmentation
        transform_params = get_params(self.opt, img_size)
        img_transform = get_transform(self.opt, transform_params)

        data = get_data(file_name_input, file_name_output, dataroot, img_transform, multiple_replace_image)

        return data

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.scenes_list)


