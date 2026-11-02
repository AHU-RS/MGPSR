import argparse
import csv
import multiprocessing as mp
import os
import random
import sys
import time
from collections import OrderedDict

import lpips
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import MGPSR_MGFDN

import common.dataset_train as dataset_train
import common.dataset_val as dataset_val
import common.utils as utils


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Stage-II")
    arg_configs = [

        (["--batch_size"], {"type": int, "default": 128, "help": "Training batch size."}),
        (["--testBatchSize"], {"type": int, "default": 16, "help": "Validation batch size."}),
        (["--n_train"], {"type": int, "default": 12092, "help": "Number of training samples."}),
        (["--n_val"], {"type": int, "default": 512, "help": "Number of samples used for validation."}),
        (["--test_every"],{"type": int,"default": 2000,"help": "Parameter used to determine the training dataset repetition factor."}),
        (["--val_every"],{"type": int, "default": 5, "help": "Validation interval in epochs."}),
        (["-nEpochs", "--nEpochs"],{"dest": "nEpochs","type": int,"default": 500,"help": "Total number of training epochs."}),
        (["--phase"],{"type": str,"default": "train","help": "Dataset phase: train or test."}),
        (["--ext"], {"type": str, "default": ".npy", "help": "file extension"}),

        (["--structure_channels"],{"type": int,"nargs": 5,"default": [24, 32, 48, 64, 96],
                "metavar": ("G1", "G2", "G3", "G4", "G5"),
                "help": "DEM and vegetation branch channels at five scales."}),
        (["--context_channels"],{"type": int, "default": 32, "help": "Legacy context channels for checkpoint compatibility."}),

        (["--lr"], {"type": float, "default": 1e-4, "help": "Initial learning rate."}),
        (["--step_size"],{"type": int, "default": 50, "help": "Learning rate decay interval in epochs."}),
        (["--gamma"], {"type": float, "default": 0.5, "help": "Learning rate decay factor."}),
        (["--seed"], {"type": int, "default": 1, "help": "Random seed."}),

        (["--root"], {"type": str, "required": True}),
        (["--train_hr_folder"], {"type": str, "required": True}),
        (["--train_lr_folder"],{"type": str, "required": True},),
        (["--train_dem_folder"], {"type": str, "required": True}),
        (["--train_ndvi_folder"], {"type": str, "required": True}),
        (["--train_evi_folder"], {"type": str, "required": True}),

        (["--val_hr_path"],{"type": str, "required": True}),
        (["--val_lr_path"],{"type": str,"required": True}),
        (["--val_dem_path"], {"type": str, "required": True}),
        (["--val_ndvi_path"],{"type": str, "required": True}),
        (["--val_evi_path"], {"type": str, "required": True}),

        (["--output_root"], {"type": str, "required": True}),
        
        (["--pretrained"],{"type": str, "default": "", "help": "Optional pretrained checkpoint; optimizer state is also restored if present."}),
        (["--resume"],{"type": str, "default": "", "help": "Optional checkpoint for resuming training."}),
        (["--start-epoch"],{"type": int,"default": 1,"help": "Starting epoch when no CSV training history is available."}),

        (["--scale"], {"type": int, "default": 3}),
        (["--patch_size"], {"type": int, "default": 96}),
        (["--n_colors"], {"type": int, "default": 1, "help": "Number of LST channels."}),
        (["--global_max"],{"type": float, "default": 316.1318, "help": "Global maximum LST value."}),
        (["--global_min"],{"type": float, "default": 229.2945, "help": "Global minimum LST value."}),
        (["--dem_max"],{"type": float, "default": 6546.0, "help": "Maximum DEM value for normalization."}),
        (["--dem_min"],{"type": float, "default": -153.0, "help": "Minimum DEM value for normalization."}),
        (["--ndvi_max"],{"type": float, "default": 1.0, "help": "Maximum NDVI value for normalization."}),
        (["--ndvi_min"],{"type": float, "default": -0.3129, "help": "Minimum NDVI value for normalization."}),
        (["--evi_max"],{"type": float, "default": 0.8791, "help": "Maximum EVI value for normalization."}),
        (["--evi_min"],{"type": float,"default": -0.2750,"help": "Minimum EVI value for normalization."},),

        (["--cuda"],{"action": argparse.BooleanOptionalAction,"default": True,"help": "Use CUDA when available; pass --no-cuda to use the CPU."}),
        (["--threads"],{"type": int, "default": 4, "help": "Number of DataLoader workers."}),

        (["--edge_loss_weight"],{"type": float, "default": 0.5, "help": "Target weight of the Sobel edge loss."}),
        (["--lap_loss_weight"],{"type": float,"default": 1.5,"help": "Target weight of the Laplacian structure loss."}),
        (["--use_physics_warmup"],{"action": "store_true","default": False,"help": "Gradually increase Sobel and Laplacian loss weights; disabled by default."}),
        (["--physics_warmup_epochs"],{"type": int,"default": 50,"help": "Number of structure-loss warmup epochs; used only when warmup is enabled."})]

    for args_name, kwargs in arg_configs:
        parser.add_argument(*args_name, **kwargs)

    return parser


class StructureGuidedReconstructionLoss(nn.Module):
    """Sobel and Laplacian losses weighted by normalized auxiliary gradients."""

    def __init__(self) -> None:
        super().__init__()

        sobel_x = torch.tensor(
            [
                [-1.0, 0.0, 1.0],
                [-2.0, 0.0, 2.0],
                [-1.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [
                [-1.0, -2.0, -1.0],
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 1.0],
            ],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        laplacian = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, -8.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.register_buffer("laplacian", laplacian)

        self.l1 = nn.L1Loss()

    def _get_normalized_gradient_mask(
        self,
        tensor_data: torch.Tensor,
    ) -> torch.Tensor:
        grad_x = F.conv2d(
            tensor_data,
            self.sobel_x,
            padding=1,
        )
        grad_y = F.conv2d(
            tensor_data,
            self.sobel_y,
            padding=1,
        )

        magnitude = torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)

        batch_size = magnitude.shape[0]
        flat = magnitude.reshape(batch_size, -1)

        value_min = flat.min(
            dim=1,
            keepdim=True,
        )[0].view(batch_size, 1, 1, 1)

        value_max = flat.max(
            dim=1,
            keepdim=True,
        )[0].view(batch_size, 1, 1, 1)

        return (magnitude - value_min) / (value_max - value_min + 1e-8)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        dem: torch.Tensor,
        ndvi: torch.Tensor,
        evi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask_dem = self._get_normalized_gradient_mask(dem)
        mask_ndvi = self._get_normalized_gradient_mask(ndvi)
        mask_evi = self._get_normalized_gradient_mask(evi)

        weight_edge = torch.maximum(
            mask_dem,
            torch.maximum(mask_ndvi, mask_evi),
        )
        weight_flat = 1.0 - weight_edge

        pred_gx = F.conv2d(pred, self.sobel_x, padding=1)
        pred_gy = F.conv2d(pred, self.sobel_y, padding=1)
        pred_mag = torch.sqrt(pred_gx.square() + pred_gy.square() + 1e-6)

        target_gx = F.conv2d(target, self.sobel_x, padding=1)
        target_gy = F.conv2d(target, self.sobel_y, padding=1)
        target_mag = torch.sqrt(target_gx.square() + target_gy.square() + 1e-6)

        loss_edge = self.l1(
            pred_mag * weight_edge,
            target_mag * weight_edge,
        )

        pred_lap = F.conv2d(
            pred,
            self.laplacian,
            padding=1,
        )
        target_lap = F.conv2d(
            target,
            self.laplacian,
            padding=1,
        )

        loss_lap = self.l1(
            pred_lap * weight_flat,
            target_lap * weight_flat,
        )

        return loss_edge, loss_lap


def seed_everything(seed: int) -> int:
    print("Random Seed:", seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return seed


def resolve_num_workers(requested_workers: int) -> int:
    if requested_workers < 0:
        raise ValueError("--threads must be non-negative.")

    max_workers = mp.cpu_count()

    if requested_workers > max_workers:
        print(
            f"===> threads={requested_workers} exceeds the number of CPU cores: "
            f"{max_workers}; using {max_workers}."
        )

    return min(requested_workers, max_workers)


def seed_worker(worker_id: int) -> None:
    del worker_id

    worker_seed = torch.initial_seed() % (2**32)

    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_dataloaders(
    args: argparse.Namespace,
    cuda_enabled: bool,
    num_workers: int,
    train_generator: torch.Generator,
) -> tuple[DataLoader, DataLoader]:
    print("===> Loading datasets")

    trainset = dataset_train.rs(args)

    valset = dataset_val.DatasetFromFolderVal(
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

    total_val = len(valset)
    n_val = min(args.n_val, total_val)

    if n_val <= 0:
        raise ValueError("The number of validation samples must be positive.")

    if n_val < total_val:
        val_rng = random.Random(args.seed)
        val_indices = sorted(val_rng.sample(range(total_val), n_val))
        eval_dataset = Subset(valset, val_indices)

        print(
            f"===> Validation subset: "
            f"{len(val_indices)} / {total_val}, seed={args.seed}"
        )
    else:
        eval_dataset = valset

        print(f"===> Validation uses full set: {total_val} samples")

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": cuda_enabled,
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["worker_init_fn"] = seed_worker

    training_loader = DataLoader(
        dataset=trainset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=train_generator,
        **loader_kwargs,
    )

    validation_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=args.testBatchSize,
        shuffle=False,
        **loader_kwargs,
    )

    return training_loader, validation_loader


def build_model_and_optimizer(
    args: argparse.Namespace,
    device: torch.device,
):
    print("===> Building MGFDN")

    model = MGPSR_MGFDN(
        n_channels=4,
        res_down=True,
        n_resblocks=1,
        bilinear=False,
        structure_channels=args.structure_channels,
        context_channels=args.context_channels,
    ).to(device)

    l1_criterion = nn.L1Loss().to(device)
    l2_criterion = nn.MSELoss().to(device)

    structure_criterion = StructureGuidedReconstructionLoss().to(device)

    print("===> Initializing LPIPS model")
    lpips_fn = lpips.LPIPS(net="squeeze").to(device)
    lpips_fn.eval()

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    amp_enabled = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        device.type,
        enabled=amp_enabled,
    )

    return (
        model,
        l1_criterion,
        l2_criterion,
        structure_criterion,
        lpips_fn,
        optimizer,
        amp_enabled,
        scaler,
    )


def _safe_torch_load(
    path: str,
    device: torch.device,
):
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=device,
        )


def load_checkpoint_if_needed(
    model: nn.Module,
    optimizer: optim.Optimizer,
    load_path: str,
    device: torch.device,
) -> None:
    """Load model weights and restore optimizer state when available."""

    if not load_path:
        return

    if not os.path.isfile(load_path):
        raise FileNotFoundError(f"Checkpoint not found: {load_path}")

    print(f"===> Loading checkpoint: {load_path}")

    checkpoint = _safe_torch_load(load_path, device)

    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )

    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        new_state_dict[new_key] = value

    try:
        model.load_state_dict(
            new_state_dict,
            strict=True,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint does not match the current MGFDN structure. Check "
            "that structure_channels and context_channels match the training configuration, "
            "and use a checkpoint from the same architecture without a residual correction head."
        ) from exc

    print("===> Model weights loaded successfully")

    if isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("===> Optimizer state loaded successfully")


def save_checkpoint(
    args: argparse.Namespace,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
) -> None:
    model_folder = os.path.join(
        args.output_root,
        f"checkpoint_x{args.scale}",
    )
    os.makedirs(model_folder, exist_ok=True)

    model_path = os.path.join(
        model_folder,
        f"epoch_{epoch}.pth",
    )

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": {
            "residual_correction": "none",
            "structure_channels": list(args.structure_channels),
            "context_channels": args.context_channels,
        },
    }

    torch.save(state, model_path)

    print(f"===> Checkpoint saved: {model_path}")


def get_structure_loss_weights(
    epoch: int,
    args: argparse.Namespace,
) -> tuple[float, float]:
    """Optionally warm up the Sobel and Laplacian loss weights."""

    if not args.use_physics_warmup:
        return (
            args.edge_loss_weight,
            args.lap_loss_weight,
        )

    if args.physics_warmup_epochs <= 0:
        ratio = 1.0
    else:
        ratio = min(
            epoch / args.physics_warmup_epochs,
            1.0,
        )

    return (
        args.edge_loss_weight * ratio,
        args.lap_loss_weight * ratio,
    )


def train_one_epoch(
    epoch: int,
    args: argparse.Namespace,
    device: torch.device,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
    l1_criterion: nn.Module,
    l2_criterion: nn.Module,
    structure_criterion: nn.Module,
    training_loader: DataLoader,
    writer: SummaryWriter,
) -> float:
    model.train()

    utils.adjust_learning_rate(
        optimizer,
        epoch,
        args.step_size,
        args.lr,
        args.gamma,
    )

    edge_weight, lap_weight = get_structure_loss_weights(epoch, args)

    print(
        f"Epoch={epoch} | "
        f"lr={optimizer.param_groups[0]['lr']:.8f} | "
        f"edge_w={edge_weight:.4f} | "
        f"lap_w={lap_weight:.4f}"
    )

    epoch_loss = 0.0

    for iteration, batch in enumerate(
        training_loader,
        start=1,
    ):
        lr_tensor, hr_tensor, dem_tensor, ndvi_tensor, evi_tensor = [
            tensor.to(device, non_blocking=args.cuda) for tensor in batch
        ]

        input_tensor = torch.cat(
            [
                lr_tensor,
                dem_tensor,
                ndvi_tensor,
                evi_tensor,
            ],
            dim=1,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            sr_tensor = model(input_tensor)

            loss_l1 = l1_criterion(
                sr_tensor,
                hr_tensor,
            )
            loss_l2 = l2_criterion(
                sr_tensor,
                hr_tensor,
            )

            loss_edge, loss_lap = structure_criterion(
                sr_tensor,
                hr_tensor,
                dem_tensor,
                ndvi_tensor,
                evi_tensor,
            )

            total_loss = (2.0 * loss_l1+ 1.0 * loss_l2+ edge_weight * loss_edge+ lap_weight * loss_lap)

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += total_loss.item()

        if iteration % 200 == 0:
            print(
                f"===> Epoch[{epoch}] "
                f"({iteration}/{len(training_loader)}) | "
                f"Total={total_loss.item():.5f} | "
                f"L1={loss_l1.item():.5f} | "
                f"L2={loss_l2.item():.5f} | "
                f"Sobel={loss_edge.item():.5f} | "
                f"Lap={loss_lap.item():.5f}"
            )

            global_step = (epoch - 1) * len(training_loader) + iteration

            writer.add_scalar(
                "Loss/Total",
                total_loss.item(),
                global_step,
            )
            writer.add_scalar(
                "Loss/L1",
                loss_l1.item(),
                global_step,
            )
            writer.add_scalar(
                "Loss/L2",
                loss_l2.item(),
                global_step,
            )
            writer.add_scalar(
                "Loss/Sobel",
                loss_edge.item(),
                global_step,
            )
            writer.add_scalar(
                "Loss/Laplacian",
                loss_lap.item(),
                global_step,
            )

    return epoch_loss / len(training_loader)


def _prepare_lpips_input(
    x: torch.Tensor,
) -> torch.Tensor:
    """Convert normalized single-band LST to three-channel LPIPS input."""

    x = x.float() * 2.0 - 1.0

    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)

    return x


def validate(
    args: argparse.Namespace,
    device: torch.device,
    model: nn.Module,
    lpips_fn: nn.Module,
    validation_loader: DataLoader,
    amp_enabled: bool,
) -> tuple[float, float, float, float]:
    """Evaluate predictions after cropping scale pixels from each border."""

    model.eval()

    rmse_sum = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0
    sample_count = 0

    with torch.no_grad():
        for batch in validation_loader:
            lr_tensor, hr_tensor, dem_tensor, ndvi_tensor, evi_tensor = [
                tensor.to(device, non_blocking=args.cuda) for tensor in batch
            ]

            input_tensor = torch.cat(
                [
                    lr_tensor,
                    dem_tensor,
                    ndvi_tensor,
                    evi_tensor,
                ],
                dim=1,
            )

            with torch.amp.autocast(
                device_type=device.type,
                enabled=amp_enabled,
            ):
                prediction = model(input_tensor)

            crop_size = args.scale

            pred_cropped = prediction[
                ...,
                crop_size:-crop_size,
                crop_size:-crop_size,
            ]
            hr_cropped = hr_tensor[
                ...,
                crop_size:-crop_size,
                crop_size:-crop_size,
            ]

            batch_lpips = lpips_fn(
                _prepare_lpips_input(pred_cropped),
                _prepare_lpips_input(hr_cropped),
            ).reshape(-1)

            for index in range(prediction.size(0)):
                sr_img = utils.tensor2np(
                    prediction[index],
                    args.global_max,
                    args.global_min,
                )
                gt_img = utils.tensor2np(
                    hr_tensor[index],
                    args.global_max,
                    args.global_min,
                )

                sr_img = utils.shave(
                    sr_img,
                    crop_size,
                )
                gt_img = utils.shave(
                    gt_img,
                    crop_size,
                )

                metrics = utils.calculate_metrics(
                    sr_img,
                    gt_img,
                    global_max=args.global_max,
                    global_min=args.global_min,
                )

                rmse_sum += metrics[0]
                psnr_sum += metrics[1]
                ssim_sum += metrics[2]
                lpips_sum += batch_lpips[index].item()

                sample_count += 1

    if sample_count == 0:
        raise RuntimeError("The validation set is empty; metrics cannot be computed.")

    avg_rmse = rmse_sum / sample_count
    avg_psnr = psnr_sum / sample_count
    avg_ssim = ssim_sum / sample_count
    avg_lpips = lpips_sum / sample_count

    print(
        "===> Valid | "
        f"RMSE={avg_rmse:.4f} | "
        f"PSNR={avg_psnr:.4f} | "
        f"SSIM={avg_ssim:.4f} | "
        f"LPIPS={avg_lpips:.4f}"
    )

    return (
        avg_rmse,
        avg_psnr,
        avg_ssim,
        avg_lpips,
    )


def print_network(net: nn.Module) -> None:
    num_params = sum(
        parameter.numel() for parameter in net.parameters() if parameter.requires_grad
    )

    print(f"Trainable parameters: {num_params:,}")


def prepare_log_file(
    args: argparse.Namespace,
):
    os.makedirs(args.output_root, exist_ok=True)

    csv_path = os.path.join(
        args.output_root,
        f"training_log_x{args.scale}.csv",
    )

    columns = [
        "Epoch",
        "Time_Seconds",
        "Train_Loss",
        "Valid_RMSE",
        "Valid_PSNR",
        "Valid_SSIM",
        "Valid_LPIPS",
    ]

    start_epoch = args.start_epoch
    write_header = not os.path.exists(csv_path)

    if not write_header:
        try:
            history = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            write_header = True
            print(f"===> Empty CSV log; initializing header: {csv_path}")
        else:
            if len(history) > 0:
                start_epoch = int(history["Epoch"].iloc[-1]) + 1

    csv_file = open(
        csv_path,
        mode="w" if write_header else "a",
        newline="",
        encoding="utf-8-sig",
    )
    csv_writer = csv.writer(csv_file)

    if write_header:
        csv_writer.writerow(columns)

        csv_file.flush()

    return (
        csv_path,
        start_epoch,
        csv_file,
        csv_writer,
    )


def main() -> None:
    args = build_parser().parse_args()

    print(args)

    torch.backends.cudnn.benchmark = True

    args.seed = seed_everything(args.seed)

    cuda_enabled = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda_enabled else "cpu")
    args.cuda = cuda_enabled

    print(f"===> Device: {device}")

    num_workers = resolve_num_workers(args.threads)

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)

    training_loader, validation_loader = build_dataloaders(
        args,
        cuda_enabled,
        num_workers,
        train_generator,
    )

    (
        model,
        l1_criterion,
        l2_criterion,
        structure_criterion,
        lpips_fn,
        optimizer,
        amp_enabled,
        scaler,
    ) = build_model_and_optimizer(
        args,
        device,
    )

    load_path = args.resume if args.resume else args.pretrained

    load_checkpoint_if_needed(
        model,
        optimizer,
        load_path,
        device,
    )

    tensorboard_dir = os.path.join(
        args.output_root,
        "tensorboard",
    )
    writer = SummaryWriter(log_dir=tensorboard_dir)

    print("===> Residual correction: none | output = LST_base + residual")

    if args.use_physics_warmup:
        print(
            f"===> Structure-loss warmup enabled | epochs={args.physics_warmup_epochs}"
        )

    print_network(model)

    (
        csv_path,
        start_epoch,
        csv_file,
        csv_writer,
    ) = prepare_log_file(args)

    try:
        for epoch in range(
            start_epoch,
            args.nEpochs + 1,
        ):
            epoch_start = time.time()

            avg_loss = train_one_epoch(
                epoch=epoch,
                args=args,
                device=device,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                amp_enabled=amp_enabled,
                l1_criterion=l1_criterion,
                l2_criterion=l2_criterion,
                structure_criterion=structure_criterion,
                training_loader=training_loader,
                writer=writer,
            )

            if epoch % args.val_every == 0 or epoch == args.nEpochs:
                (
                    avg_rmse,
                    avg_psnr,
                    avg_ssim,
                    avg_lpips,
                ) = validate(
                    args=args,
                    device=device,
                    model=model,
                    lpips_fn=lpips_fn,
                    validation_loader=validation_loader,
                    amp_enabled=amp_enabled,
                )
            else:
                avg_rmse = float("nan")
                avg_psnr = float("nan")
                avg_ssim = float("nan")
                avg_lpips = float("nan")

                print(
                    f"===> Epoch {epoch}: skip validation; val_every={args.val_every}"
                )

            save_checkpoint(
                args,
                epoch,
                model,
                optimizer,
            )

            elapsed = time.time() - epoch_start

            csv_writer.writerow(
                [
                    epoch,
                    round(elapsed, 2),
                    round(avg_loss, 6),
                    round(avg_rmse, 6),
                    round(avg_psnr, 6),
                    round(avg_ssim, 6),
                    round(avg_lpips, 6),
                ]
            )
            csv_file.flush()

    finally:
        writer.close()
        csv_file.close()

    print(f"===> Training finished. CSV log: {csv_path}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
