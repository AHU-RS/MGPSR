import argparse
import csv
import os
import sys
import time
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import common.utils as utils
from model import MGPSR_MGFDN


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test Stage-II")

    parser.add_argument("--test_hr_folder",type=str,default=None)
    parser.add_argument("--test_lr_folder",type=str,required=True)
    parser.add_argument("--test_dem_folder",type=str,required=True)
    parser.add_argument("--test_ndvi_folder",type=str,required=True)
    parser.add_argument("--test_evi_folder",type=str,required=True)

    parser.add_argument("--output_folder",type=str,required=True,)
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint.",
    )

    parser.add_argument("--structure_channels",type=int,nargs=5,default=[24, 32, 48, 64, 96],
                        metavar=("G1", "G2", "G3", "G4", "G5"),help="Auxiliary branch channels at five scales.")
    parser.add_argument("--context_channels",type=int,default=32,help="Legacy context channels for checkpoint compatibility.")

    parser.add_argument("--cuda",action=argparse.BooleanOptionalAction,default=True,help="Use CUDA when available; pass --no-cuda to use the CPU.")
    parser.add_argument("--upscale_factor",type=int,default=3,help="Nominal scale factor and border width cropped during evaluation.")
    parser.add_argument("--save_aux_maps",action=argparse.BooleanOptionalAction,default=True,help="Save residual arrays; pass --no-save_aux_maps to disable.")

    parser.add_argument("--global_max",type=float,default=316.1318,help="Global maximum LST value.")
    parser.add_argument("--global_min",type=float,default=229.2945,help="Global minimum LST value.")
    parser.add_argument("--dem_max",type=float,default=6546.0,help="Maximum DEM value.")
    parser.add_argument("--dem_min",type=float,default=-153.0,help="Minimum DEM value.")
    parser.add_argument("--ndvi_max",type=float,default=1.0,help="Maximum NDVI value.")
    parser.add_argument("--ndvi_min",type=float,default=-0.3129,help="Minimum NDVI value.")
    parser.add_argument("--evi_max",type=float,default=0.8791,help="Maximum EVI value.")
    parser.add_argument("--evi_min",type=float,default=-0.2750,help="Minimum EVI value.")

    return parser


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


def load_model(
    args: argparse.Namespace,
    device: torch.device,
) -> MGPSR_MGFDN:
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    print(f"===> Loading checkpoint: {args.checkpoint}")
    checkpoint = _safe_torch_load(args.checkpoint, device)

    config = checkpoint.get("model_config", {}) if isinstance(checkpoint, dict) else {}
    if not isinstance(config, dict):
        raise RuntimeError("The checkpoint model_config must be a dictionary.")

    saved_variant = config.get("residual_correction")
    if saved_variant is not None and saved_variant != "none":
        raise RuntimeError(
            f"This script requires residual_correction=none; "
            f"the checkpoint specifies {saved_variant!r}. Use matching model code and checkpoint weights."
        )

    for name in ("structure_channels", "context_channels"):
        if name not in config:
            continue
        saved_value = config[name]
        requested_value = getattr(args, name)
        if name == "structure_channels":
            saved_value = tuple(saved_value)
            requested_value = tuple(requested_value)
        if saved_value != requested_value:
            raise RuntimeError(
                f"Structural parameter {name} differs: checkpoint={saved_value},"
                f"requested={requested_value}. Use the training configuration for --{name}."
            )

    model = MGPSR_MGFDN(
        n_channels=4,
        res_down=True,
        n_resblocks=1,
        bilinear=False,
        structure_channels=args.structure_channels,
        context_channels=args.context_channels,
    ).to(device)

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
        model.load_state_dict(new_state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint does not match the current model without a residual correction head. "
            "Use weights trained with the same model and structural settings."
        ) from exc

    model.eval()
    return model


def read_single_band(
    path: str,
    name: str,
) -> np.ndarray:
    image = cv2.imread(
        path,
        cv2.IMREAD_ANYDEPTH,
    )

    if image is None:
        raise FileNotFoundError(
            f"{name} file is missing or unreadable: {path}"
        )

    return image


def normalize_to_tensor(
    image: np.ndarray,
    value_min: float,
    value_max: float,
    device: torch.device,
) -> torch.Tensor:
    """Normalize a single-band image and return a [1, 1, H, W] tensor."""

    if value_max <= value_min:
        raise ValueError(
            "Invalid normalization range: value_max must be greater than value_min."
        )

    if image.ndim == 2:
        image = image[..., np.newaxis]

    normalized = (
        (image.astype(np.float32) - value_min)
        / (value_max - value_min)
    )

    tensor = torch.from_numpy(
        np.transpose(normalized, (2, 0, 1))
    ).unsqueeze(0)

    return tensor.float().to(device)


def match_spatial_size(
    tensor: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Match array sizes; inputs must already be geographically registered."""

    if tensor.shape[-2:] == reference.shape[-2:]:
        return tensor

    return F.interpolate(
        tensor,
        size=reference.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )


def save_auxiliary_arrays(
    aux: dict[str, torch.Tensor],
    output_folder: str,
    base_name: str,
) -> None:
    """Save normalized predicted residuals."""
    aux_dir = os.path.join(output_folder, "aux_maps")
    os.makedirs(aux_dir, exist_ok=True)
    residual = aux["residual"].detach().float().cpu().numpy()[0]
    np.save(os.path.join(aux_dir, f"{base_name}_residual.npy"), residual)


def main() -> None:
    args = build_parser().parse_args()

    print(args)

    cuda_enabled = (
        args.cuda
        and torch.cuda.is_available()
    )
    device = torch.device(
        "cuda"
        if cuda_enabled
        else "cpu"
    )

    print(f"===> Device: {device}")

    model = load_model(
        args,
        device,
    )

    os.makedirs(
        args.output_folder,
        exist_ok=True,
    )

    use_hr = args.test_hr_folder is not None
    if use_hr and not os.path.isdir(args.test_hr_folder):
        raise FileNotFoundError(
            f"HR reference directory does not exist: {args.test_hr_folder}"
        )

    filelist = utils.get_list(
        args.test_lr_folder,
        ext=".tif",
    )

    if len(filelist) == 0:
        raise RuntimeError(
            f"No TIFF files found in the Stage-I LST directory: {args.test_lr_folder}"
        )

    num_samples = len(filelist)

    rmse_list = np.full(
        num_samples,
        np.nan,
        dtype=np.float64,
    )
    psnr_list = np.full(
        num_samples,
        np.nan,
        dtype=np.float64,
    )
    ssim_list = np.full(
        num_samples,
        np.nan,
        dtype=np.float64,
    )
    time_list = np.zeros(
        num_samples,
        dtype=np.float64,
    )
    per_image_rows = []

    if cuda_enabled:
        start_event = torch.cuda.Event(
            enable_timing=True
        )
        end_event = torch.cuda.Event(
            enable_timing=True
        )
    else:
        start_event = None
        end_event = None

    for index, lr_path in enumerate(filelist):
        base_name = os.path.splitext(
            os.path.basename(lr_path)
        )[0]

        print(
            f"[{index + 1}/{num_samples}] {base_name}"
        )

        hr = None
        if use_hr:
            hr_path = os.path.join(
                args.test_hr_folder,
                base_name + ".tif",
            )
            hr = read_single_band(
                hr_path,
                "MODIS reference",
            )
            hr = utils.modcrop(
                hr,
                args.upscale_factor,
            )

        dem_path = os.path.join(
            args.test_dem_folder,
            base_name + ".tif",
        )
        ndvi_path = os.path.join(
            args.test_ndvi_folder,
            base_name + ".tif",
        )
        evi_path = os.path.join(
            args.test_evi_folder,
            base_name + ".tif",
        )

        stage1_lst = read_single_band(
            lr_path,
            "Stage-I LST",
        )
        dem = read_single_band(
            dem_path,
            "DEM",
        )
        ndvi = read_single_band(
            ndvi_path,
            "NDVI",
        )
        evi = read_single_band(
            evi_path,
            "EVI",
        )

        lst_tensor = normalize_to_tensor(
            stage1_lst,
            args.global_min,
            args.global_max,
            device,
        )
        dem_tensor = normalize_to_tensor(
            dem,
            args.dem_min,
            args.dem_max,
            device,
        )
        ndvi_tensor = normalize_to_tensor(
            ndvi,
            args.ndvi_min,
            args.ndvi_max,
            device,
        )
        evi_tensor = normalize_to_tensor(
            evi,
            args.evi_min,
            args.evi_max,
            device,
        )

        dem_tensor = match_spatial_size(
            dem_tensor,
            lst_tensor,
        )
        ndvi_tensor = match_spatial_size(
            ndvi_tensor,
            lst_tensor,
        )
        evi_tensor = match_spatial_size(
            evi_tensor,
            lst_tensor,
        )

        input_tensor = torch.cat(
            [
                lst_tensor,
                dem_tensor,
                ndvi_tensor,
                evi_tensor,
            ],
            dim=1,
        )

        with torch.no_grad():
            if cuda_enabled:
                start_event.record()
            else:
                start_time = time.perf_counter()

            if args.save_aux_maps:
                prediction, aux = model(
                    input_tensor,
                    return_aux=True,
                )
            else:
                prediction = model(input_tensor)
                aux = None

            if cuda_enabled:
                end_event.record()
                torch.cuda.synchronize()

                time_ms = start_event.elapsed_time(
                    end_event
                )
            else:
                time_ms = (
                    time.perf_counter()
                    - start_time
                ) * 1000.0

        time_list[index] = time_ms

        output_image = utils.tensor2np(
            prediction[0],
            args.global_max,
            args.global_min,
        )

        rmse = float("nan")
        psnr = float("nan")
        ssim = float("nan")

        if hr is not None:
            crop_size = args.upscale_factor

            sr_eval = utils.shave(
                output_image,
                crop_size,
            )
            hr_eval = utils.shave(
                hr,
                crop_size,
            )

            metrics = utils.calculate_metrics(
                sr_eval,
                hr_eval,
                global_max=args.global_max,
                global_min=args.global_min,
            )

            rmse, psnr, ssim = metrics

            rmse_list[index] = rmse
            psnr_list[index] = psnr
            ssim_list[index] = ssim

        output_path = os.path.join(
            args.output_folder,
            base_name + ".tif",
        )

        # OpenCV output does not retain GeoTIFF georeferencing.
        if not cv2.imwrite(
            output_path,
            output_image,
        ):
            raise IOError(
                f"Failed to save prediction: {output_path}"
            )

        if (
            args.save_aux_maps
            and aux is not None
        ):
            save_auxiliary_arrays(
                aux,
                args.output_folder,
                base_name,
            )

        per_image_rows.append(
            [
                base_name,
                rmse,
                psnr,
                ssim,
                time_ms,
            ]
        )

        if use_hr:
            message = (
                f"    RMSE={rmse:.4f} | "
                f"PSNR={psnr:.4f} | "
                f"SSIM={ssim:.4f} | "
                f"TIME={time_ms:.2f} ms"
            )
        else:
            message = (
                f"    Inference only | TIME={time_ms:.2f} ms"
            )

        print(message)

    metrics_csv = os.path.join(
        args.output_folder,
        "test_metrics.csv",
    )

    with open(
        metrics_csv,
        mode="w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "Name",
                "RMSE",
                "PSNR",
                "SSIM",
                "Time_ms",
            ]
        )

        writer.writerows(per_image_rows)

    print("=" * 72)
    if use_hr:
        print("===> Final Average Results")
        print(
            f"Samples: {num_samples} | "
            f"RMSE: {np.mean(rmse_list):.4f} | "
            f"PSNR: {np.mean(psnr_list):.4f} | "
            f"SSIM: {np.mean(ssim_list):.4f} | "
            f"TIME: {np.mean(time_list):.2f} ms"
        )
    else:
        print("===> Inference Completed (HR not provided)")
        print(
            f"Samples: {num_samples} | "
            f"TIME: {np.mean(time_list):.2f} ms"
        )

    print(f"Per-image metrics: {metrics_csv}")
    print("=" * 72)


if __name__ == "__main__":
    main()
