import torch
import torch.utils.data as data
from os.path import join
from os import listdir
from torchvision.transforms import Compose, ToTensor
from PIL import Image
import numpy as np
from torchvision import transforms
import cv2

def img_modcrop(image, modulo):
    # sz ： (H, W) or (H, W, C)
    sz = image.shape 
    h = (sz[0] // modulo) * modulo
    w = (sz[1] // modulo) * modulo
    
    if image.ndim == 2:
        return image[0:h, 0:w]
    return image[0:h, 0:w, :]

def custom_to_tensor(img,global_max, global_min):
    img_np = np.array(img, dtype=np.float32)
    if img_np.ndim == 2:
        img_np = np.expand_dims(img_np, axis=0)  

    tensor = torch.from_numpy(img_np)
    data_range = global_max - global_min
    tensor.sub_(global_min).div_(data_range)

    return tensor


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".bmp", ".png", ".tif"])


def load_image(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise ValueError(f"Unable to read the image: {filepath}")
    return img

class DatasetFromFolderVal(data.Dataset):
    def __init__(self, hr_dir, lr_dir, dem_dir, ndvi_dir, evi_dir,
                  upscale, global_max, global_min, dem_max, dem_min, ndvi_max, ndvi_min, evi_max, evi_min):
        super(DatasetFromFolderVal, self).__init__()
        self.hr_filenames = sorted([join(hr_dir, x) for x in listdir(hr_dir) if is_image_file(x)])
        self.lr_filenames = sorted([join(lr_dir, x) for x in listdir(lr_dir) if is_image_file(x)])
        self.dem_filenames = sorted([join(dem_dir, x) for x in listdir(dem_dir) if is_image_file(x)])
        self.ndvi_filenames = sorted([join(ndvi_dir, x) for x in listdir(ndvi_dir) if is_image_file(x)])
        self.evi_filenames = sorted([join(evi_dir, x) for x in listdir(evi_dir) if is_image_file(x)])
        self.upscale = upscale
        self.global_max = global_max
        self.global_min = global_min
        self.dem_max = dem_max
        self.dem_min = dem_min
        self.ndvi_max = ndvi_max
        self.ndvi_min = ndvi_min
        self.evi_max = evi_max
        self.evi_min = evi_min

    def __getitem__(self, index):
        input_img = load_image(self.lr_filenames[index])
        target_img  = load_image(self.hr_filenames[index])
        dem_img = load_image(self.dem_filenames[index])
        ndvi_img = load_image(self.ndvi_filenames[index])
        evi_img = load_image(self.evi_filenames[index])
        input_tensor = custom_to_tensor(input_img, self.global_max, self.global_min)
        target_tensor = custom_to_tensor(img_modcrop(target_img, self.upscale), self.global_max, self.global_min)
        dem_tensor = custom_to_tensor(dem_img, self.dem_max, self.dem_min)
        ndvi_tensor = custom_to_tensor(ndvi_img, self.ndvi_max, self.ndvi_min)
        evi_tensor = custom_to_tensor(evi_img, self.evi_max, self.evi_min)
        return input_tensor, target_tensor, dem_tensor, ndvi_tensor, evi_tensor

    def __len__(self):
        return len(self.lr_filenames)
