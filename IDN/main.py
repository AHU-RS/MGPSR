import argparse
import csv
import multiprocessing as mp
import os
import random
import sys
import time
from collections import OrderedDict

import lpips
import pandas as pd
import skimage.color as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Subset

# Add the project root (Upload) to the module search path.
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
import common.dataset_train as dataset_train
import common.dataset_val as dataset_val
import common.utils as utils
from model import MGPSR_IDN

parser = argparse.ArgumentParser(description="MGPSR_IDN training")
arg_configs = [
    (["--batch_size"], {"type": int, "default": 128, "help": "training batch size"}),
    (["--testBatchSize"], {"type": int, "default": 16, "help": "testing batch size"}),
    (["-nEpochs"], {"type": int, "default": 500, "help": "number of epochs to train"}),
    (["--lr"], {"type": float, "default": 0.0001, "help": "Learning Rate. Default=0.0001"}),
    (["--step_size"], {"type": int, "default": 50, "help": "learning rate decay per N epochs"}),
    (["--gamma"], {"type": float, "default": 0.5, "help": "learning rate decay factor for step decay"}),
    (["--n_train"], {"type": int, "default": 12092, "help": "number of training set"}),
    (["--n_val"], {"type": int, "default": 1620, "help": "number of validation set"}),
    (["--test_every"], {"type": int, "default": 2000, "help": "test frequency"}),

    (["--root"], {"type": str, "default": r" ", "help": "dataset directory"}),
    (["--train_hr_folder"], {"type": str, "default": "MODIS_3km_npy", "help": "training HR images folder name"}),
    (["--train_lr_folder"], {"type": str, "default": "CLDAS_3km_npy", "help": "training LR images folder name"}),
    (["--train_dem_folder"], {"type": str, "default": "DEM_3km_npy", "help": "training DEM folder name"}),
    (["--train_ndvi_folder"], {"type": str, "default": "NDVI_3km_npy", "help": "training NDVI folder name"}),
    (["--train_evi_folder"], {"type": str, "default": "EVI_3km_npy", "help": "training EVI folder name"}),
    
    (["--val_hr_path"], {"type": str, "default": r"\MODIS_3km", "help": "validation HR images directory"}),
    (["--val_lr_path"], {"type": str, "default": r"\CLDAS_3km", "help": "validation LR images directory"}),
    (["--val_dem_path"], {"type": str, "default": r"\DEM_3km", "help": "validation DEM images directory"}),
    (["--val_ndvi_path"], {"type": str, "default": r"\NDVI_3km", "help": "validation NDVI directory"}),
    (["--val_evi_path"], {"type": str, "default": r"\EVI_3km", "help": "validation EVI directory"}),
    
    (["--output_root"], {"type": str, "default": r"D:\PRO-TESR\S2", "help": "root directory for saving checkpoints and logs"}),

    
    
    (["--pretrained"], {"type": str, "default": "", "help": "path to pretrained models"}),
    (["--resume"], {"type": str, "default": r"", "help": "path to checkpoint"}),
    (["--scale"], {"type": int, "default": 2, "help": "super-resolution scale"}),
    (["--patch_size"], {"type": int, "default": 32, "help": "output patch size"}), 
    (["--rgb_range"], {"type": int, "default": 1, "help": "maxium value of RGB"}),
    (["--n_colors"], {"type": int, "default": 1, "help": "number of color channels to use"}),
    (["--ext"], {"type": str, "default": ".npy", "help": "file extension"}),
    (["--isY"], {"action": "store_true", "default": False, "help": "evaluate on Y channel"}),
    (["--global_max"], {"type": float, "default": 316.1318, "help": "dataset max value"}),
    (["--global_min"], {"type": float, "default": 229.2945, "help": "dataset min value"}),
    (["--dem_max"], {"type": float, "default": 6546.0, "help": "DEM max value"}),
    (["--dem_min"], {"type": float, "default": -153.0, "help": "DEM min value"}),
    (["--evi_max"], {"type": float, "default": 0.8791, "help": "EVI max value"}),
    (["--evi_min"], {"type": float, "default": -0.2750, "help": "EVI min value"}),
    (["--ndvi_max"], {"type": float, "default": 1.0, "help": "NDVI max value"}),
    (["--ndvi_min"], {"type": float, "default": -0.3129, "help": "NDVI min value"}),
    (["--cuda"], {"action": "store_true", "default": True, "help": "use cuda"}),
    (["--start-epoch"], {"type": int, "default": 1, "help": "manual epoch number"}),
    (["--threads"], {"type": int, "default": 4, "help": "number of workers for data loading"}),
    (["--seed"], {"type": int, "default": 1, "help": "random seed"}),
    (["--phase"], {"type": str, "default": "train", "help": "train or test phase"}),
]
for args_name, kwargs in arg_configs:
    parser.add_argument(*args_name, **kwargs)


args = None
cuda = False
device = None
model = None
l1_criterion = None
l2_criterion = None
physics_criterion = None
lpips_fn = None
optimizer = None
writer = None
training_data_loader = None
testing_data_loader = None


class SpatialStructuralLoss(nn.Module):
    def __init__(self, device):
        super(SpatialStructuralLoss, self).__init__()
        self.sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3).to(device)
        self.sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3).to(device)
        self.laplacian = torch.tensor(
            [[1.0, 1.0, 1.0], [1.0, -8.0, 1.0], [1.0, 1.0, 1.0]]
        ).view(1, 1, 3, 3).to(device)
        self.l1 = nn.L1Loss()

    def _get_normalized_gradient_mask(self, tensor_data):
        grad_x = F.conv2d(tensor_data, self.sobel_x, padding=1)
        grad_y = F.conv2d(tensor_data, self.sobel_y, padding=1)
        slope = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

        b, _, _, _ = slope.shape
        slope_view = slope.view(b, -1)
        slope_min = slope_view.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        slope_max = slope_view.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        mask = (slope - slope_min) / (slope_max - slope_min + 1e-8)
        return mask

    def forward(self, pred, target, dem, ndvi, evi):
        mask_dem = self._get_normalized_gradient_mask(dem)
        mask_ndvi = self._get_normalized_gradient_mask(ndvi)
        mask_evi = self._get_normalized_gradient_mask(evi)

        weight_edge = torch.max(mask_dem, torch.max(mask_ndvi, mask_evi))
        weight_flat = 1.0 - weight_edge

        pred_gx = F.conv2d(pred, self.sobel_x, padding=1)
        pred_gy = F.conv2d(pred, self.sobel_y, padding=1)
        pred_mag = torch.sqrt(pred_gx ** 2 + pred_gy ** 2 + 1e-6)

        target_gx = F.conv2d(target, self.sobel_x, padding=1)
        target_gy = F.conv2d(target, self.sobel_y, padding=1)
        target_mag = torch.sqrt(target_gx ** 2 + target_gy ** 2 + 1e-6)

        loss_edge = self.l1(pred_mag * weight_edge, target_mag * weight_edge)

        pred_lap = F.conv2d(pred, self.laplacian, padding=1)
        target_lap = F.conv2d(target, self.laplacian, padding=1)
        loss_lap = self.l1(pred_lap * weight_flat, target_lap * weight_flat)

        return loss_edge, loss_lap


def build_val_loader(testset, args):
    total_val = len(testset)
    n_val = min(args.n_val, total_val)
    if n_val <= 0:
        raise ValueError("n_val must be positive.")

    if n_val < total_val:
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        indices = torch.randperm(total_val, generator=generator)[:n_val].tolist()
        val_dataset = Subset(testset, indices)
    else:
        val_dataset = testset

    loader = DataLoader(
        dataset=val_dataset,
        num_workers=args.threads,
        batch_size=args.testBatchSize,
        shuffle=False,
        pin_memory=args.cuda,
        drop_last=False,
        persistent_workers=args.threads > 0,
    )
    return loader, n_val, total_val


def train(epoch):
    model.train()
    utils.adjust_learning_rate(optimizer, epoch, args.step_size, args.lr, args.gamma)
    print("epoch =", epoch, "lr =", optimizer.param_groups[0]["lr"])

    epoch_loss = 0.0
    global_step_base = (epoch - 1) * len(training_data_loader)

    for iteration, (lr_tensor, hr_tensor, dem_tensor, ndvi_tensor, evi_tensor) in enumerate(training_data_loader, 1):
        if args.cuda:
            lr_tensor = lr_tensor.to(device, non_blocking=True)
            hr_tensor = hr_tensor.to(device, non_blocking=True)
            dem_tensor = dem_tensor.to(device, non_blocking=True)
            ndvi_tensor = ndvi_tensor.to(device, non_blocking=True)
            evi_tensor = evi_tensor.to(device, non_blocking=True)

        input_tensor = torch.cat([lr_tensor, dem_tensor, ndvi_tensor, evi_tensor], dim=1)

        optimizer.zero_grad()
        sr_tensor = model(input_tensor)

        loss_l1 = l1_criterion(sr_tensor, hr_tensor)
        loss_l2 = l2_criterion(sr_tensor, hr_tensor)
        loss_edge, loss_lap = physics_criterion(sr_tensor, hr_tensor, dem_tensor, ndvi_tensor, evi_tensor)
        loss_sr = 2.0 * loss_l1 + 1.0 * loss_l2 + 0.5 * loss_edge + 1.5 * loss_lap

        loss_sr.backward()
        optimizer.step()

        epoch_loss += loss_sr.item()

        if iteration % 200 == 0:
            print(
                "===> Epoch[{}]({}/{}): Total Loss: {:.5f} | L1: {:.5f} | L2: {:.5f} | Edge: {:.5f} | Laplacian: {:.5f}".format(
                    epoch,
                    iteration,
                    len(training_data_loader),
                    loss_sr.item(),
                    loss_l1.item(),
                    loss_l2.item(),
                    loss_edge.item(),
                    loss_lap.item(),
                )
            )
            step = global_step_base + iteration
            writer.add_scalar("Loss/Total", loss_sr.item(), step)
            writer.add_scalar("Loss/L1", loss_l1.item(), step)
            writer.add_scalar("Loss/Edge", loss_edge.item(), step)
            writer.add_scalar("Loss/Laplacian", loss_lap.item(), step)

    avg_epoch_loss = epoch_loss / len(training_data_loader)
    return avg_epoch_loss


def valid():
    model.eval()
    rmse_sum, psnr_sum, ssim_sum, lpips_sum = 0.0, 0.0, 0.0, 0.0
    sample_count = 0
    crop_size = args.scale

    with torch.no_grad():
        for lr_tensor, hr_tensor, dem_tensor, ndvi_tensor, evi_tensor in testing_data_loader:
            if args.cuda:
                lr_tensor = lr_tensor.to(device, non_blocking=True)
                hr_tensor = hr_tensor.to(device, non_blocking=True)
                dem_tensor = dem_tensor.to(device, non_blocking=True)
                ndvi_tensor = ndvi_tensor.to(device, non_blocking=True)
                evi_tensor = evi_tensor.to(device, non_blocking=True)

            input_tensor = torch.cat([lr_tensor, dem_tensor, ndvi_tensor, evi_tensor], dim=1)
            pre = model(input_tensor)

            pre_cropped = pre[..., crop_size:-crop_size, crop_size:-crop_size]
            hr_cropped = hr_tensor[..., crop_size:-crop_size, crop_size:-crop_size]
            pre_lpips = pre_cropped * 2.0 - 1.0
            hr_lpips = hr_cropped * 2.0 - 1.0
            batch_lpips = lpips_fn(pre_lpips, hr_lpips).view(-1)

            for i in range(pre.size(0)):
                sr_img = utils.tensor2np(pre[i].detach(), args.global_max, args.global_min)
                gt_img = utils.tensor2np(hr_tensor[i].detach(), args.global_max, args.global_min)

                cropped_sr_img = utils.shave(sr_img, crop_size)
                cropped_gt_img = utils.shave(gt_img, crop_size)

                if args.isY:
                    im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
                    im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
                else:
                    im_label = cropped_gt_img
                    im_pre = cropped_sr_img

                rmse_value, psnr_value, ssim_value = utils.calculate_metrics(
                    im_pre,
                    im_label,
                    global_max=args.global_max,
                    global_min=args.global_min,
                )
                rmse_sum += rmse_value
                psnr_sum += psnr_value
                ssim_sum += ssim_value
                lpips_sum += batch_lpips[i].item()
                sample_count += 1

    if sample_count == 0:
        raise RuntimeError("Validation loader returned zero samples.")

    avg_rmse = rmse_sum / sample_count
    avg_psnr = psnr_sum / sample_count
    avg_ssim = ssim_sum / sample_count
    avg_lpips = lpips_sum / sample_count

    print(
        "===> Valid:  RMSE: {:.4f}, PSNR: {:.4f}, SSIM: {:.4f}, LPIPS: {:.4f}".format(
            avg_rmse, avg_psnr, avg_ssim, avg_lpips
        )
    )
    return avg_rmse, avg_psnr, avg_ssim, avg_lpips


def save_checkpoint(epoch):
    model_folder = os.path.join(args.output_root, "checkpoint_x{}".format(args.scale))
    model_out_path = os.path.join(model_folder, "epoch_{}.pth".format(epoch))
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(state, model_out_path)
    print("===> Checkpoint saved to {}".format(model_out_path))


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print("Total number of parameters: %d" % num_params)


def main():
    global args, cuda, device
    global model, l1_criterion, l2_criterion, physics_criterion, lpips_fn, optimizer, writer
    global training_data_loader, testing_data_loader

    args = parser.parse_args()
    print(args)

    torch.backends.cudnn.benchmark = True

    seed = args.seed
    if seed is None:
        seed = random.randint(1, 10000)
    print("Ramdom Seed: ", seed)
    random.seed(seed)
    torch.manual_seed(seed)

    cuda = args.cuda
    device = torch.device("cuda" if cuda else "cpu")
    args.is_train = True

    print("===> Loading datasets")
    trainset = dataset_train.rs(args)
    testset = dataset_val.DatasetFromFolderVal(
        args.val_hr_path,
        args.val_lr_path,
        args.val_dem_path,
        args.val_ndvi_path,
        args.val_evi_path,
        args.scale,
        args.global_max,
        args.global_min,
        args.dem_max,
        args.dem_min,
        args.ndvi_max,
        args.ndvi_min,
        args.evi_max,
        args.evi_min,
    )

    training_data_loader = DataLoader(
        dataset=trainset,
        num_workers=args.threads,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=args.cuda,
        drop_last=True,
        persistent_workers=args.threads > 0,
    )
    testing_data_loader, actual_n_val, total_val = build_val_loader(testset, args)
    print(f"===> Validation subset: {actual_n_val}/{total_val}, testBatchSize={args.testBatchSize}")

    print("===> Building models")
    model = MGPSR_IDN(n_channels=4, res_down=True, n_resblocks=1, bilinear=0)

    l1_criterion = nn.L1Loss()
    l2_criterion = nn.MSELoss()
    print("===> Initializing multi-source guided loss...")
    physics_criterion = SpatialStructuralLoss(device)

    print("===> Initializing LPIPS model...")
    lpips_fn = lpips.LPIPS(net="squeeze").to(device)
    lpips_fn.eval()

    print("===> Setting GPU")
    if cuda:
        model = model.to(device)
        l1_criterion = l1_criterion.to(device)
        l2_criterion = l2_criterion.to(device)

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    load_path = args.resume if args.resume else args.pretrained
    if load_path:
        if os.path.isfile(load_path):
            print(f"===> Loading checkpoint '{load_path}'")
            checkpoint = torch.load(load_path, map_location=device)

            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith("module.") else k
                new_state_dict[name] = v

            model.load_state_dict(new_state_dict, strict=True)
            print("===> Model weights loaded successfully.")

            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                print("===> Optimizer state loaded successfully. (Seamless resume ready)")
        else:
            print(f"===> No checkpoint found at '{load_path}'")

    writer = SummaryWriter()

    print("===> Training")
    print_network(model)
    os.makedirs(args.output_root, exist_ok=True)
    csv_filename = os.path.join(args.output_root, f"training_log_x{args.scale}.csv")

    if os.path.exists(csv_filename):
        df_log = pd.read_csv(csv_filename)
        if len(df_log) > 0:
            start_epoch = int(df_log["Epoch"].iloc[-1]) + 1
        else:
            start_epoch = args.start_epoch
        csv_file = open(csv_filename, mode="a", newline="")
        csv_writer = csv.writer(csv_file)
    else:
        start_epoch = args.start_epoch
        csv_file = open(csv_filename, mode="w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Epoch", "Time_Seconds", "Train_Loss", "Valid_RMSE", "Valid_PSNR", "Valid_SSIM", "Valid_LPIPS"])

    try:
        for epoch in range(start_epoch, args.nEpochs + 1):
            epoch_start_time = time.time()

            avg_loss = train(epoch)
            avg_rmse, avg_psnr, avg_ssim, avg_lpips = valid()
            save_checkpoint(epoch)

            epoch_duration = time.time() - epoch_start_time
            csv_writer.writerow(
                [
                    epoch,
                    round(epoch_duration, 2),
                    round(avg_loss, 4),
                    round(avg_rmse, 4),
                    round(avg_psnr, 4),
                    round(avg_ssim, 4),
                    round(avg_lpips, 4),
                ]
            )
            csv_file.flush()
    finally:
        csv_file.close()
        writer.close()

    print(f"===> Training finished. Log saved to {csv_filename}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
