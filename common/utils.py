import lpips
import numpy as np
import os,math,cv2
import torch
from collections import OrderedDict

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def shave(im, border):
    border = [border, border]
    im = im[border[0]:-border[0], border[1]:-border[1], ...]
    return im


def modcrop(im, modulo):
    sz = im.shape

    h = np.int32(sz[0] / modulo) * modulo
    w = np.int32(sz[1] / modulo) * modulo
    ims = im[0:h, 0:w, ...]
    return ims


def get_list(path, ext):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(ext)]


def convert_shape(img):
    img = np.transpose((img * 255.0).round(), (1, 2, 0))
    img = np.uint8(np.clip(img, 0, 255))
    return img


def quantize(img):
    return img.clip(0, 255)
    # return img.clip(0, 255).round().astype(np.uint8)

def tensor2np(tensor, global_max, global_min, out_type=np.float32):
    tensor = tensor.float().cpu()
    data_range = global_max - global_min
    
    img_np = tensor.numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    
    if out_type == np.float32:
        img_np = (img_np * data_range + global_min).round(5)
        return img_np.astype(out_type)
    

    return img_np.astype(out_type)


def convert2np(tensor):
    return tensor.cpu().mul(255).clamp(0, 255).byte().squeeze().permute(1, 2, 0).numpy()


def adjust_learning_rate(optimizer, epoch, step_size, lr_init, gamma):
    factor = epoch // step_size
    lr = lr_init * (gamma ** factor)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def load_state_dict(path):

    state_dict = torch.load(path)
    new_state_dcit = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            name = k[7:]
        else:
            name = k
        new_state_dcit[name] = v
    return new_state_dcit


def _squeeze_singleton_channel(image):
    if image.ndim == 3 and image.shape[-1] == 1:
        return image[..., 0]
    return image

def calculate_metrics(im_pre, im_label, global_max, global_min):

    im_label = _squeeze_singleton_channel(im_label)
    im_pre = _squeeze_singleton_channel(im_pre)

    if im_label.ndim != im_pre.ndim:
        raise ValueError(
            f" label={im_label.shape}, pred={im_pre.shape}"
        )

    min_h = min(im_label.shape[0], im_pre.shape[0])
    min_w = min(im_label.shape[1], im_pre.shape[1])
    im_label = im_label[:min_h, :min_w]
    im_pre = im_pre[:min_h, :min_w]

    im_label = im_label.astype(np.float32)
    im_pre = im_pre.astype(np.float32)

    # 1. RMSE
    rmse = np.sqrt(((im_label - im_pre) ** 2).mean())

    # 2. PSNR 
    fixed_data_range = global_max - global_min
    psnr_value = psnr(im_label, im_pre, data_range=fixed_data_range)

    # 3. SSIM
    min_side = min(im_label.shape[0], im_label.shape[1])
    win_size = min(7, min_side)  
    if win_size % 2 == 0:
        win_size -= 1
        
    if win_size < 3:
        ssim_value = float('nan')
    else:
        if im_label.ndim == 3:
            ssim_value, _ = ssim(im_label, im_pre, full=True, data_range=fixed_data_range, win_size=win_size, channel_axis=-1)
        else:
            ssim_value, _ = ssim(im_label, im_pre, full=True, data_range=fixed_data_range, win_size=win_size)

    return rmse, psnr_value, ssim_value

_global_lpips_model = None

def calculate_lpips_tensor(tensor_pre, tensor_label, device):
    global _global_lpips_model
    if _global_lpips_model is None:
        _global_lpips_model = lpips.LPIPS(net='squeeze').to(device)
        _global_lpips_model.eval()
        for param in _global_lpips_model.parameters():
            param.requires_grad = False

    t_pre = tensor_pre.clamp(0.0, 1.0) * 2.0 - 1.0
    t_label = tensor_label.clamp(0.0, 1.0) * 2.0 - 1.0

    if t_pre.shape[1] == 1:
        t_pre = t_pre.repeat(1, 3, 1, 1)
        t_label = t_label.repeat(1, 3, 1, 1)

    with torch.no_grad():
        dist = _global_lpips_model(t_pre, t_label)
        
    return dist.mean().item()
