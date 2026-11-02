import argparse
import os
import sys
import time
from collections import OrderedDict

import cv2
import numpy as np
import skimage.color as sc
import torch
import torch.nn.functional as F

# Add the project root (Upload) to the module search path.
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import common.utils as utils
from model import MGPSR_IDN


parser = argparse.ArgumentParser(description="MGPSR_IDN testing/inference")

# parser.add_argument("--test_hr_folder",type=str,default=None,
#                     help="optional folder of target HR images; omit for inference-only mode",) 
# parser.add_argument("--test_lr_folder",type=str,default=r" ",help="folder of input LR images",)
# parser.add_argument("--test_dem_folder",type=str,default=r" ",help="folder of DEM images",)
# parser.add_argument("--test_ndvi_folder",type=str,default=r" ",help="folder of NDVI images",)
# parser.add_argument("--test_evi_folder",type=str,default=r" ",help="folder of EVI images",)
# parser.add_argument("--output_folder",type=str,default=r" ",help="folder in which SR images are saved",)
# parser.add_argument("--checkpoint",type=str,default=r" ",help="checkpoint file",)
parser.add_argument("--test_hr_folder",type=str,default=None,
                    help="optional folder of target HR images; omit for inference-only mode",)
parser.add_argument("--test_lr_folder",type=str,default=r" ",
                    help="folder of input LR images",)
parser.add_argument("--test_dem_folder",type=str,default=r" ",
                    help="folder of DEM images",)
parser.add_argument("--test_ndvi_folder",type=str,default=r" ",
                    help="folder of NDVI images",)
parser.add_argument("--test_evi_folder",type=str,default=r" ",
                    help="folder of EVI images",)
parser.add_argument("--output_folder",type=str,default=r" ",
                    help="folder in which SR images are saved",)
parser.add_argument("--checkpoint",type=str,
                    default=r"D:\Final-Code\MGPNet\Upload\IDN\epoch_268.pth",help="checkpoint file",)

parser.add_argument("--cuda", action="store_true", default=True, help="use CUDA")
parser.add_argument("--upscale_factor", type=int, default=2, help="border crop size/scale factor")
parser.add_argument("--is_y", action="store_true", default=False, help="evaluate Y channel")

# The normalization range must be exactly the same as that used during the training phase.
parser.add_argument("--global_max", type=float, default=316.1318, help="maximum LST value")
parser.add_argument("--global_min", type=float, default=229.2945, help="minimum LST value")
parser.add_argument("--dem_max", type=float, default=6546.0, help="maximum DEM value")
parser.add_argument("--dem_min", type=float, default=-153.0, help="minimum DEM value")
parser.add_argument("--evi_max", type=float, default=0.8791, help="maximum EVI value")
parser.add_argument("--evi_min", type=float, default=-0.2750, help="minimum EVI value")
parser.add_argument("--ndvi_max", type=float, default=1.0, help="maximum NDVI value")
parser.add_argument("--ndvi_min", type=float, default=-0.3129, help="minimum NDVI value")


def list_tif_files(folder):
    """Return a deterministic LR-driven sample list, accepting .tif and .TIF."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    files = [
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if os.path.splitext(name)[1].lower() in {".tif", ".tiff"}
    ]
    files.sort()
    if not files:
        raise FileNotFoundError(f"No TIFF files found in: {folder}")
    return files


def find_matching_tif(folder, base_name):
    """Find an auxiliary image with the same stem and a common TIFF extension."""
    for suffix in (".tif", ".TIF", ".tiff", ".TIFF"):
        path = os.path.join(folder, base_name + suffix)
        if os.path.isfile(path):
            return path
    return os.path.join(folder, base_name + ".tif")


def read_single_band(path, name):
    image = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    if image is None:
        raise FileNotFoundError(f"{name} image is missing or unreadable: {path}")
    if image.ndim == 2:
        image = image[..., np.newaxis]
    return image


def to_normalized_tensor(image, value_min, value_max):
    if value_max <= value_min:
        raise ValueError(f"Invalid normalization range: min={value_min}, max={value_max}")
    array = (image.astype(np.float32) - value_min) / (value_max - value_min)
    array = np.transpose(array, (2, 0, 1))[np.newaxis, ...]
    return torch.from_numpy(array).float()


opt = parser.parse_args()
print(opt)

use_cuda = opt.cuda and torch.cuda.is_available()
if opt.cuda and not torch.cuda.is_available():
    print("Warning: CUDA was requested but is unavailable; falling back to CPU.")
device = torch.device("cuda" if use_cuda else "cpu")

filelist = list_tif_files(opt.test_lr_folder)
use_hr = opt.test_hr_folder is not None
if use_hr and not os.path.isdir(opt.test_hr_folder):
    raise FileNotFoundError(f"HR folder does not exist: {opt.test_hr_folder}")
for folder_name, folder in (
    ("DEM", opt.test_dem_folder),
    ("NDVI", opt.test_ndvi_folder),
    ("EVI", opt.test_evi_folder),
):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"{folder_name} folder does not exist: {folder}")
os.makedirs(opt.output_folder, exist_ok=True)

print("===> Mode: validation with HR" if use_hr else "===> Mode: inference only (no HR)")

model = MGPSR_IDN(n_channels=4, res_down=True, n_resblocks=1, bilinear=0)
print(f"===> Loading checkpoint {opt.checkpoint}")
# checkpoint = torch.load(opt.checkpoint, map_location=device)
# state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
# new_state_dict = OrderedDict(
#     (key[7:] if key.startswith("module.") else key, value)
#     for key, value in state_dict.items()
# )
# model.load_state_dict(new_state_dict, strict=True)
checkpoint = torch.load(
    opt.checkpoint,
    map_location=device,
    weights_only=True,
)

state_dict = (
    checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
    else checkpoint
)

new_state_dict = OrderedDict()

for old_name, value in state_dict.items():
    name = old_name

    # Remove the DataParallel prefix.
    if name.startswith("module."):
        name = name[7:]

    # DoubleConv: double_conv -> block.
    name = name.replace(".double_conv.", ".block.")

    # SpatialAttention: conv1 -> conv.
    parts = name.split(".")
    if (
        len(parts) >= 3
        and parts[0] in {"sa1", "sa2", "sa3", "sa4"}
        and parts[1] == "conv1"
    ):
        parts[1] = "conv"
        name = ".".join(parts)

    if name in new_state_dict:
        raise RuntimeError(
            f"Duplicate parameter name after conversion: {name}"
        )

    new_state_dict[name] = value

model.load_state_dict(new_state_dict, strict=True)
print("===> Checkpoint loaded successfully after parameter-name conversion.")
model = model.to(device)
model.eval()

rmse_list = []
psnr_list = []
ssim_list = []
time_list = []

for index, lr_path in enumerate(filelist, start=1):
    base_name = os.path.splitext(os.path.basename(lr_path))[0]

    im_l = read_single_band(lr_path, "LR")
    im_dem = read_single_band(find_matching_tif(opt.test_dem_folder, base_name), "DEM")
    im_ndvi = read_single_band(find_matching_tif(opt.test_ndvi_folder, base_name), "NDVI")
    im_evi = read_single_band(find_matching_tif(opt.test_evi_folder, base_name), "EVI")

    im_gt = None
    if use_hr:
        hr_path = find_matching_tif(opt.test_hr_folder, base_name)
        im_gt = read_single_band(hr_path, "HR")
        im_gt = utils.modcrop(im_gt, opt.upscale_factor)

    input_lr = to_normalized_tensor(im_l, opt.global_min, opt.global_max).to(device)
    input_dem = to_normalized_tensor(im_dem, opt.dem_min, opt.dem_max).to(device)
    input_ndvi = to_normalized_tensor(im_ndvi, opt.ndvi_min, opt.ndvi_max).to(device)
    input_evi = to_normalized_tensor(im_evi, opt.evi_min, opt.evi_max).to(device)

    with torch.no_grad():
        target_size = input_lr.shape[-2:]
        if input_dem.shape[-2:] != target_size:
            input_dem = F.interpolate(input_dem, size=target_size, mode="bilinear", align_corners=False)
        if input_ndvi.shape[-2:] != target_size:
            input_ndvi = F.interpolate(input_ndvi, size=target_size, mode="bilinear", align_corners=False)
        if input_evi.shape[-2:] != target_size:
            input_evi = F.interpolate(input_evi, size=target_size, mode="bilinear", align_corners=False)

        model_input = torch.cat([input_lr, input_dem, input_ndvi, input_evi], dim=1)
        if use_cuda:
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        output = model(model_input)
        if use_cuda:
            torch.cuda.synchronize()
        time_list.append((time.perf_counter() - start_time) * 1000.0)

    out_img = utils.tensor2np(output.detach()[0], opt.global_max, opt.global_min)
    output_path = os.path.join(opt.output_folder, base_name + ".tif")
    if not cv2.imwrite(output_path, out_img):
        raise OSError(f"Failed to write SR image: {output_path}")

    metrics = None
    if im_gt is not None:
        crop_size = opt.upscale_factor
        cropped_sr = utils.shave(out_img, crop_size)
        cropped_gt = utils.shave(im_gt, crop_size)
        if opt.is_y:
            if cropped_sr.shape[-1] != 3 or cropped_gt.shape[-1] != 3:
                raise ValueError("--is_y requires three-channel SR and HR images")
            im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr)[:, :, 0])
            im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt)[:, :, 0])
        else:
            im_pre = cropped_sr
            im_label = cropped_gt

        metrics = utils.calculate_metrics(
            im_pre,
            im_label,
            global_max=opt.global_max,
            global_min=opt.global_min,
        )
        rmse_list.append(metrics[0])
        psnr_list.append(metrics[1])
        ssim_list.append(metrics[2])

    if metrics is None:
        print(f"[{index}/{len(filelist)}] Processed {base_name} - inference only")
    else:
        print(
            f"[{index}/{len(filelist)}] Processed {base_name} - "
            f"RMSE: {metrics[0]:.4f}, PSNR: {metrics[1]:.4f}, SSIM: {metrics[2]:.4f}"
        )

print("=" * 60)
if rmse_list:
    print("===> Final Average Validation Results <===")
    print(
        "Mean RMSE: {:.4f} | Mean PSNR: {:.4f} | Mean SSIM: {:.4f} | "
        "Mean TIME: {:.2f} ms".format(
            np.mean(rmse_list),
            np.mean(psnr_list),
            np.mean(ssim_list),
            np.mean(time_list),
        )
    )
else:
    print("===> Inference Completed (metrics skipped because HR was not provided) <===")
    print(
        "Processed: {} | Mean TIME: {:.2f} ms | Output: {}".format(
            len(filelist), np.mean(time_list), opt.output_folder
        )
    )
print("=" * 60)
