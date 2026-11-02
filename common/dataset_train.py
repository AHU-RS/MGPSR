import torch.utils.data as data
import os.path
import cv2
import numpy as np
try:
    from . import dataset_transforms
except ImportError:
    import dataset_transforms

def default_loader(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.ndim == 3:
        return img[:, :, [2, 1, 0]]
    return img

def npy_loader(path):
    return np.load(path)

IMG_EXTENSIONS = [
    '.tif', '.npy',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


class rs(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.scale = self.opt.scale
        self.root = self.opt.root
        self.ext = self.opt.ext
        self.train = True if self.opt.phase == 'train' else False
        self.repeat = self.opt.test_every // (self.opt.n_train // self.opt.batch_size)
        self._set_filesystem(self.root)
        self.images_hr, self.images_lr, self.images_dem, self.images_ndvi, self.images_evi = self._scan()

    def _set_filesystem(self, dir_path):
        self.dir_hr = os.path.join(dir_path, self.opt.train_hr_folder)
        self.dir_lr = os.path.join(dir_path, self.opt.train_lr_folder)
        self.dir_dem = os.path.join(dir_path, self.opt.train_dem_folder)
        self.dir_ndvi = os.path.join(dir_path, self.opt.train_ndvi_folder) 
        self.dir_evi = os.path.join(dir_path, self.opt.train_evi_folder)   


    def __getitem__(self, idx):
        lr, hr, dem ,ndvi, evi= self._load_file(idx)
        lr, hr, dem, ndvi, evi = self._get_patch(lr, hr, dem, ndvi, evi) 
        lr, hr, dem, ndvi, evi = dataset_transforms.set_channel(lr, hr, dem, ndvi, evi, n_channels=self.opt.n_colors)
        
        lr_tensor, hr_tensor, dem_tensor, ndvi_tensor, evi_tensor = dataset_transforms.np2Tensor(
            lr, hr, dem, ndvi, evi,
            global_max=self.opt.global_max, 
            global_min=self.opt.global_min,
            dem_max=self.opt.dem_max,    
            dem_min=self.opt.dem_min,
            ndvi_max=self.opt.ndvi_max,
            ndvi_min=self.opt.ndvi_min,
            evi_max=self.opt.evi_max,
            evi_min=self.opt.evi_min
        ) 
        return lr_tensor, hr_tensor, dem_tensor, ndvi_tensor, evi_tensor

    def __len__(self):
        if self.train:
            return self.opt.n_train * self.repeat

    def _get_index(self, idx):
        if self.train:
            return idx % self.opt.n_train
        else:
            return idx

    def _get_patch(self, img_in, img_tar, img_dem, img_ndvi, img_evi):
        patch_size = self.opt.patch_size
        scale = self.scale
        if self.train:
            img_in, img_tar, img_dem, img_ndvi, img_evi = dataset_transforms.get_patch(
                img_in, img_tar, img_dem, img_ndvi, img_evi, patch_size=patch_size, scale=scale)
            img_in, img_tar, img_dem, img_ndvi, img_evi = dataset_transforms.augment(img_in, img_tar, img_dem, img_ndvi, img_evi)
        else:
            ih, iw = img_in.shape[:2]
            img_tar = img_tar[0:ih * scale, 0:iw * scale, :]
            img_dem = img_dem[0:ih, 0:iw] if img_dem.ndim == 2 else img_dem[0:ih, 0:iw, :]
            img_ndvi = img_ndvi[0:ih, 0:iw] if img_ndvi.ndim == 2 else img_ndvi[0:ih, 0:iw, :]
            img_evi = img_evi[0:ih, 0:iw] if img_evi.ndim == 2 else img_evi[0:ih, 0:iw, :]
        return img_in, img_tar, img_dem, img_ndvi, img_evi

    def _scan(self):
        list_hr = sorted(make_dataset(self.dir_hr))
        list_lr = sorted(make_dataset(self.dir_lr))
        list_dem = sorted(make_dataset(self.dir_dem))
        list_ndvi = sorted(make_dataset(self.dir_ndvi)) 
        list_evi = sorted(make_dataset(self.dir_evi))   
        return list_hr, list_lr, list_dem, list_ndvi, list_evi

    def _load_file(self, idx):
        idx = self._get_index(idx)
        
        def load_by_ext(path):
            if path.endswith('.npy'):
                return npy_loader(path)
            else:
                return default_loader(path)

        lr = load_by_ext(self.images_lr[idx])
        hr = load_by_ext(self.images_hr[idx])
        dem = load_by_ext(self.images_dem[idx])
        ndvi = load_by_ext(self.images_ndvi[idx])
        evi = load_by_ext(self.images_evi[idx])
        return lr, hr, dem, ndvi, evi
