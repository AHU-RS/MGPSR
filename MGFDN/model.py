from typing import Sequence

import torch
import torch.nn as nn
from common.blocks import (
    DoubleConv,
    AuxiliaryEncoderBlock,
    StructuralContextBlock,
    SpatialAttention,
    PixelAttention,
    Down,
    Up,
    ResnetBlock,
    OutConv,
    ContextAggregator,
)

class MGPSR_MGFDN(nn.Module):
    def __init__(
        self,
        n_channels: int = 4,
        n_classes: int = 1,
        res_down: bool = True,
        n_resblocks: int = 1,
        bilinear: bool = False,
        structure_channels: Sequence[int] = (24, 32, 48, 64, 96),
        context_channels: int = 32,
    ) -> None:
        super().__init__()

        if n_channels != 4:
            raise ValueError(
                "MGFDN currently uses 4 input channels exclusively:"
                "[IDN SR LST, DEM, NDVI, EVI]."
            )

        structure_channels = tuple(structure_channels)

        if len(structure_channels) != 5:
            raise ValueError(
                "The structure_channels must specify the number of channels for 5 different scales."
            )

        g1, g2, g3, g4, g5 = structure_channels

        self.lst_stem = DoubleConv(1, 64)

        self.dem_stem = AuxiliaryEncoderBlock(1, g1)
        self.vegetation_stem = AuxiliaryEncoderBlock(2, g1)

        self.context1 = StructuralContextBlock(
            channels=64,
            aux_channels=g1 * 2,
        )
        self.sa1 = SpatialAttention()

        self.down1 = Down(64, 128, res_down=res_down)
        self.dem_down1 = AuxiliaryEncoderBlock(
            g1,
            g2,
            downsample=True,
        )
        self.vegetation_down1 = AuxiliaryEncoderBlock(
            g1,
            g2,
            downsample=True,
        )
        self.context2 = StructuralContextBlock(
            channels=128,
            aux_channels=g2 * 2,
        )
        self.sa2 = SpatialAttention()

        self.down2 = Down(128, 256, res_down=res_down)
        self.dem_down2 = AuxiliaryEncoderBlock(
            g2,
            g3,
            downsample=True,
        )
        self.vegetation_down2 = AuxiliaryEncoderBlock(
            g2,
            g3,
            downsample=True,
        )
        self.context3 = StructuralContextBlock(
            channels=256,
            aux_channels=g3 * 2,
        )
        self.sa3 = SpatialAttention()

        self.down3 = Down(256, 512, res_down=res_down)
        self.dem_down3 = AuxiliaryEncoderBlock(
            g3,
            g4,
            downsample=True,
        )
        self.vegetation_down3 = AuxiliaryEncoderBlock(
            g3,
            g4,
            downsample=True,
        )
        self.context4 = StructuralContextBlock(
            channels=512,
            aux_channels=g4 * 2,
        )
        self.sa4 = SpatialAttention()

        self.down4 = Down(512, 1024, res_down=res_down)
        self.dem_down4 = AuxiliaryEncoderBlock(
            g4,
            g5,
            downsample=True,
        )
        self.vegetation_down4 = AuxiliaryEncoderBlock(
            g4,
            g5,
            downsample=True,
        )
        self.context5 = StructuralContextBlock(
            channels=1024,
            aux_channels=g5 * 2,
        )

        self.pa1 = PixelAttention(1024)

        self.resblocks = nn.Sequential(
            *[
                ResnetBlock(
                    1024,
                    padding_type="reflect",
                    norm_layer=nn.BatchNorm2d,
                    use_dropout=False,
                    use_bias=True,
                )
                for _ in range(n_resblocks)
            ]
        )

        self.up1 = Up(
            in_channels=1024,
            skip_channels=512,
            out_channels=512,
            bilinear=bilinear,
        )
        self.up2 = Up(
            in_channels=512,
            skip_channels=256,
            out_channels=256,
            bilinear=bilinear,
        )
        self.up3 = Up(
            in_channels=256,
            skip_channels=128,
            out_channels=128,
            bilinear=bilinear,
        )
        self.up4 = Up(
            in_channels=128,
            skip_channels=64,
            out_channels=64,
            bilinear=bilinear,
        )

        self.pa2 = PixelAttention(64)

        self.residual_head = OutConv(64, n_classes)

        # Retained only for legacy model and optimizer checkpoint compatibility.
        self.context_aggregator = ContextAggregator(
            in_channels=(64, 128, 256, 512, 1024),
            context_channels=context_channels,
        )

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        if x.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W] input, got {x.shape}.")

        if x.shape[1] != 4:
            raise ValueError(
                "The input channel sequence for MGFDN must be "
                "[IDN SR LST, DEM, NDVI, EVI]."
            )

        lst_base = x[:, 0:1]
        dem = x[:, 1:2]
        vegetation = x[:, 2:4]

        lst1 = self.lst_stem(lst_base)
        dem1 = self.dem_stem(dem)
        vegetation1 = self.vegetation_stem(vegetation)

        x1, _ = self.context1(lst1, dem1, vegetation1)
        sa1 = self.sa1(x1)

        x2 = self.down1(x1)
        dem2 = self.dem_down1(dem1)
        vegetation2 = self.vegetation_down1(vegetation1)

        x2, _ = self.context2(x2, dem2, vegetation2)
        sa2 = self.sa2(x2)

        x3 = self.down2(x2)
        dem3 = self.dem_down2(dem2)
        vegetation3 = self.vegetation_down2(vegetation2)

        x3, _ = self.context3(x3, dem3, vegetation3)
        sa3 = self.sa3(x3)

        x4 = self.down3(x3)
        dem4 = self.dem_down3(dem3)
        vegetation4 = self.vegetation_down3(vegetation3)

        x4, _ = self.context4(x4, dem4, vegetation4)
        sa4 = self.sa4(x4)

        x5 = self.down4(x4)
        dem5 = self.dem_down4(dem4)
        vegetation5 = self.vegetation_down4(vegetation4)

        x5, _ = self.context5(x5, dem5, vegetation5)

        x5 = self.pa1(x5) + x5
        x5 = self.resblocks(x5)

        u4 = sa4 + x4
        u3 = sa3 + x3
        u2 = sa2 + x2
        u1 = sa1 + x1

        y = self.up1(x5, u4)
        y = self.up2(y, u3)
        y = self.up3(y, u2)
        y = self.up4(y, u1)

        decode_feat = self.pa2(y) + y

        residual = self.residual_head(decode_feat)

        out = lst_base + residual

        if return_aux:
            return out, {"residual": residual}

        return out
