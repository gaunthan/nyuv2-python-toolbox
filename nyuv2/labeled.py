import h5py
import numpy as np
from PIL import Image

def rotate_image(image):
    return image.rotate(-90, expand=True)

class LabeledDataset:
    """Python interface for the labeled subset of the NYU dataset.

    To save memory, call the `close()` method of this class to close
    the dataset file once you're done using it.
    """

    def __init__(self, path):
        """Opens the labeled dataset file at the given path."""
        self.file = h5py.File(path)
        self.color_maps = self.file['images']
        self.depth_maps = self.file['depths']
        self.label_maps = self.file['labels']

        self.scene_types = self.file['sceneTypes'][0]
        self.scene_names = self.file['scenes'][0]
        self.label_names = None

    def close(self):
        """Closes the HDF5 file from which the dataset is read."""
        self.file.close()

    def __len__(self):
        return len(self.color_maps)

    def __getitem__(self, idx):
        color_map = self.color_maps[idx]
        color_map = np.moveaxis(color_map, 0, -1)
        color_image = Image.fromarray(color_map, mode='RGB')
        color_image = rotate_image(color_image)

        depth_map = self.depth_maps[idx]
        depth_image = Image.fromarray(depth_map, mode='F')
        depth_image = rotate_image(depth_image)

        label_map = self.label_maps[idx]
        label_image = Image.fromarray(np.uint8(label_map))
        label_image = rotate_image(label_image)

        raw_scene_type = self.file[self.scene_types[idx]]
        scene_type = ''.join([chr(ch[0]) for ch in raw_scene_type])

        raw_scene_name = self.file[self.scene_names[idx]]
        scene_name = ''.join([chr(ch[0]) for ch in raw_scene_name])

        return color_image, depth_image, label_image, scene_type, scene_name

    def get_label_names(self):
        if not self.label_names:
            label_name_maps = self.file['names'][0]
            label_names = ['unlabeled']
            for name in label_name_maps:
                name = ''.join([chr(ch[0]) for ch in self.file[name]])
                label_names.append(name)
            self.label_names = label_names
        return self.label_names

    def get_scene_names(self):
        return self.scene_names

    def get_scene_types(self):
        return self.scene_types