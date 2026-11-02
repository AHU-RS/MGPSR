from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        res_down: bool = True,
    ) -> None:
        super().__init__()
        self.res_down = res_down

        if res_down:
            self.in_conv = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=2,
                stride=2,
            )
            self.mid_conv = DoubleConv(in_channels, in_channels)
            self.out_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.res_down:
            return self.maxpool_conv(x)

        x_down = self.in_conv(x)
        return self.out_conv(self.mid_conv(x_down) + x_down)


class Up(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        bilinear: bool = True,
    ) -> None:
        super().__init__()

        if bilinear:
            # Bilinear interpolation only alters the spatial dimensions, 
            # without changing the number of channels.
            self.up = nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=True,
            )
            self.conv = DoubleConv(
                in_channels + skip_channels,
                out_channels,
            )
        else:
            # Transposed convolution simultaneously performs upsampling and channel compression.
            self.up = nn.ConvTranspose2d(
                in_channels,
                skip_channels,
                kernel_size=2,
                stride=2,
            )
            self.conv = DoubleConv(
                skip_channels * 2,
                out_channels,
            )

    def forward(
        self,
        x_low: torch.Tensor,
        x_skip: torch.Tensor,
    ) -> torch.Tensor:
        x_low = self.up(x_low)

        # When the input size cannot be completely divided by 2^n, 
        # symmetric padding is applied to the upsampled features to ensure that 
        # their spatial dimensions are consistent with the features of the skip connections.
        diff_y = x_skip.size(2) - x_low.size(2)
        diff_x = x_skip.size(3) - x_low.size(3)

        if diff_x != 0 or diff_y != 0:
            x_low = F.pad(
                x_low,
                [
                    diff_x // 2,
                    diff_x - diff_x // 2,
                    diff_y // 2,
                    diff_y - diff_y // 2,
                ],
                mode="reflect",
            )

        return self.conv(torch.cat([x_skip, x_low], dim=1))


class OutConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResnetBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        padding_type: str = "reflect",
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        use_dropout: bool = False,
        use_bias: bool = True,
    ) -> None:
        super().__init__()

        self.conv_block = self._build_conv_block(
            dim=dim,
            padding_type=padding_type,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            use_bias=use_bias,
        )

    @staticmethod
    def _padding_layer(padding_type: str) -> Tuple[list[nn.Module], int]:
        if padding_type == "reflect":
            return [nn.ReflectionPad2d(1)], 0
        if padding_type == "replicate":
            return [nn.ReplicationPad2d(1)], 0
        if padding_type == "zero":
            return [], 1

        raise NotImplementedError(
            f"Unsupported padding_type：{padding_type}"
        )

    def _build_conv_block(
        self,
        dim: int,
        padding_type: str,
        norm_layer: type[nn.Module],
        use_dropout: bool,
        use_bias: bool,
    ) -> nn.Sequential:
        block: list[nn.Module] = []

        padding, p = self._padding_layer(padding_type)
        block.extend(padding)
        block.extend(
            [
                nn.Conv2d(
                    dim,
                    dim,
                    kernel_size=3,
                    padding=p,
                    bias=use_bias,
                ),
                norm_layer(dim),
                nn.ReLU(inplace=True),
            ]
        )

        if use_dropout:
            block.append(nn.Dropout(0.5))

        padding, p = self._padding_layer(padding_type)
        block.extend(padding)
        block.extend(
            [
                nn.Conv2d(
                    dim,
                    dim,
                    kernel_size=3,
                    padding=p,
                    bias=use_bias,
                ),
                norm_layer(dim),
            ]
        )

        return nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        return self.sigmoid(
            self.conv(torch.cat([avg_out, max_out], dim=1))
        )


class PixelAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigmoid(self.conv(x))

SA = SpatialAttention
PA = PixelAttention

class AuxiliaryEncoderBlock(nn.Module):
    """
    Vegetation/terrain coding blocks.
    When downsample=True, first use average pooling to synchronize the auxiliary features to the scale of the main branch.
    """

    def __init__(self,in_channels: int,out_channels: int,downsample: bool = False,) -> None:
        super().__init__()

        layers: list[nn.Module] = []

        if downsample:
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2))

        layers.extend(
            [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        )

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class StructuralContextBlock(nn.Module):
    """
    Structural context block.
    primary:
    The main branch features of the current scale's LST;
    dem_feat:
    The structural characteristics of the current-scale DEM;
    vegetation_feat:
    The structural characteristics of the current scale of NDVI/EVI.
    fused:
    The LST features after being injected with the structural context;
    context:
    The structural responses that truly participated in the fusion after being filtered by the main branch conditions.

    """

    def __init__(
        self,
        channels: int,
        aux_channels: int,
        reduction: int = 8,
    ) -> None:
        super().__init__()

        hidden_channels = max(channels // reduction, 16)

        # Project the features obtained by splicing the DEM with the vegetation branch onto the main branch channel count.
        self.aux_proj = nn.Sequential(
            nn.Conv2d(
                aux_channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # Generate gated weights based on the combined features of the main branch and the auxiliary branch.
        self.gate = nn.Sequential(
            nn.Conv2d(
                channels * 2,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.Sigmoid(),
        )

        self.context_refine = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        primary: torch.Tensor,
        dem_feat: torch.Tensor,
        vegetation_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        aux_feat = self.aux_proj(
            torch.cat([dem_feat, vegetation_feat], dim=1)
        )

        gate = self.gate(
            torch.cat([primary, aux_feat], dim=1)
        )

        # context represents the auxiliary structure response that has been selected under the current LST conditions.
        context = self.context_refine(aux_feat * gate)

        # Use residual injection.
        fused = primary + context

        return fused, context


class ContextAggregator(nn.Module):
    """
    The aggregated results are merely as an optional supplementary output and do not participate in the final LST prediction.
    """

    def __init__(
        self,
        in_channels: Sequence[int] = (64, 128, 256, 512, 1024),
        context_channels: int = 32,
    ) -> None:
        super().__init__()

        if len(in_channels) == 0:
            raise ValueError("in_channels must include at least one scale.")
        if context_channels <= 0:
            raise ValueError("context_channels must be a positive integer.")

        self.num_scales = len(in_channels)

        # Project each scale to the same number of channels to control the parameter quantity and memory usage.
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        context_channels,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(context_channels),
                    nn.ReLU(inplace=True),
                )
                for channels in in_channels
            ]
        )

        self.scale_logits = nn.Parameter(
            torch.zeros(self.num_scales)
        )

        self.refine = nn.Sequential(
            nn.Conv2d(
                context_channels,
                context_channels,
                kernel_size=3,
                padding=1,
                groups=context_channels,
                bias=False,
            ),
            nn.BatchNorm2d(context_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                context_channels,
                context_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(context_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        contexts: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if len(contexts) != self.num_scales:
            raise ValueError(
                f"Expected to receive  {self.num_scales} scales of context,"
                f"Actually received {len(contexts)} ."
            )

        target_size = contexts[0].shape[-2:]

        # The softmax function ensures that the scale weights are non-negative and their sum is equal to 1.
        scale_weights = torch.softmax(
            self.scale_logits,
            dim=0,
        )

        aggregated: Optional[torch.Tensor] = None

        for index, (context, projection) in enumerate(
            zip(contexts, self.projections)
        ):
            projected = projection(context)

            if projected.shape[-2:] != target_size:
                projected = F.interpolate(
                    projected,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )

            weighted = scale_weights[index] * projected

            aggregated = (
                weighted
                if aggregated is None
                else aggregated + weighted
            )

        if aggregated is None:
            raise RuntimeError("No available structural context for aggregation was obtained.")

        return self.refine(aggregated)