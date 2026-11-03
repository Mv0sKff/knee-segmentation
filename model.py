# UNet adapted from https://github.com/lext/deep-pipeline (MIT, (c) 2019 Aleksei Tiulpin); see LICENSE
import torch
import torch.nn.functional as F
from torch import nn


def conv_block_3x3(inp, out, activation="relu", normalization="BN", bias=True):
    if normalization == "BN":
        norm_layer = nn.BatchNorm2d(out)
    elif normalization == "IN":
        norm_layer = nn.InstanceNorm2d(out)
    elif normalization is None:
        norm_layer = nn.Sequential()
    else:
        raise NotImplementedError

    conv = nn.Conv2d(inp, out, kernel_size=3, padding=1, bias=bias)
    if activation == "relu":
        return nn.Sequential(conv, norm_layer, nn.ReLU(inplace=True))
    if activation == "selu":
        return nn.Sequential(conv, nn.SELU(inplace=True))
    if activation == "elu":
        return nn.Sequential(conv, nn.ELU(1, inplace=True))
    if activation is None:
        return nn.Sequential(conv, norm_layer)
    raise NotImplementedError


def conv_stack(inp, out, depth, activation, normalization):
    return nn.Sequential(
        *[
            conv_block_3x3(inp if i == 0 else out, out, activation, normalization)
            for i in range(depth)
        ]
    )


class Encoder(nn.Module):
    def __init__(self, inp, out, depth=2, activation="relu", normalization="BN"):
        super().__init__()
        self.layers = conv_stack(inp, out, depth, activation, normalization)

    def forward(self, x):
        processed = self.layers(x)
        return processed, F.max_pool2d(processed, 2, 2)


class Decoder(nn.Module):
    def __init__(
        self, inp, out, depth=2, mode="bilinear", activation="relu", normalization="BN"
    ):
        super().__init__()
        self.ups_mode = mode
        self.layers = conv_stack(inp, out, depth, activation, normalization)

    def forward(self, x_big, x):
        x_ups = F.interpolate(
            x, size=x_big.size()[-2:], mode=self.ups_mode, align_corners=True
        )
        return self.layers(torch.cat([x_ups, x_big], 1))


class UNet(nn.Module):
    """bw: base width, doubled at each level; depth: number of encoder levels."""

    def __init__(
        self,
        bw=24,
        depth=6,
        center_depth=2,
        n_inputs=1,
        n_classes=1,
        activation="relu",
        norm_encoder="BN",
        norm_decoder="BN",
        norm_center="BN",
    ):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least 2")

        self.encoders = nn.ModuleList(
            [Encoder(n_inputs, bw, activation=activation, normalization=norm_encoder)]
        )
        for level in range(2, depth + 1):
            self.encoders.append(
                Encoder(
                    bw * 2 ** (level - 2),
                    bw * 2 ** (level - 1),
                    activation=activation,
                    normalization=norm_encoder,
                )
            )

        mul_out = 2 ** (depth - 1)
        self.center = nn.Sequential(
            *[
                conv_block_3x3(bw * mul_out, bw * mul_out, activation, norm_center)
                for _ in range(center_depth)
            ]
        )

        # Ordered deepest first, matching the skip connections popped in forward().
        self.decoders = nn.ModuleList()
        for level in range(depth, 1, -1):
            mul_in = 2 ** (level - 1)
            self.decoders.append(
                Decoder(
                    2 * bw * mul_in,
                    bw * mul_in // 2,
                    activation=activation,
                    normalization=norm_decoder,
                )
            )
        self.decoders.append(
            Decoder(2 * bw, bw, activation=activation, normalization=norm_decoder)
        )

        self.mixer = nn.Conv2d(bw, n_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            convolved, x = encoder(x)
            skips.append(convolved)

        x = self.center(x)
        for decoder in self.decoders:
            x = decoder(skips.pop(), x)

        return self.mixer(x)
