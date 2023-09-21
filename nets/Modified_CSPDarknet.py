import torch
from torch import nn


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = SiLU()
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        patch_top_left = x[..., ::2, ::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), dim=1, )
        return self.conv(x)

def autopad(k, p=None, d=1):
    if d > 1:
        # actual kernel-size
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        # auto-pad
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, p=None, g=1, d=1, act="silu"):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, ksize, stride, autopad(ksize, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize=ksize, stride=stride, groups=in_channels, act=act)
        self.pconv = BaseConv(in_channels, out_channels, ksize=1, stride=1, groups=1, act=act)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)

class CSAM(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, act="silu"):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.cv = nn.Sequential(
            BaseConv(in_channels, out_channels, ksize, stride=1, act=act),
            BaseConv(in_channels, out_channels, ksize, stride=1, act=act)
        )
    def forward(self, x):
        x1 = self.avg(x)
        x2 = self.cv(x1)
        return x * x2.expand_as(x)

class SPPBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x

class SFEM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), act="silu"):
        super().__init__()
        c_ = in_channels // 2
        self.cv1 = BaseConv(in_channels, c_, 1, stride=1, act=act)
        self.cv2 = BaseConv(in_channels, in_channels, 3, stride=1, act=act)
        self.cv3 = BaseConv(c_ * 4, c_, 1, stride=1, act=act)
        self.cv4 = BaseConv(c_ * 2, out_channels, 1, stride=1, act=act)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv1(self.cv2(x))
        x3 = torch.cat([x2] + [m(x2) for m in self.m], dim=1)
        y  = self.cv3(x3)
        output = self.cv4(torch.cat((x1, y), dim=1))
        return output

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, ksize=3, e=0.5, act="silu"):
        super().__init__()
        c_ = int(out_channels * e)  # hidden channels
        self.cv1 = BaseConv(in_channels, c_, ksize=ksize, stride=1, act=act)
        self.cv2 = BaseConv(c_, out_channels, ksize=ksize, stride=1, act=act)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class CSPAM(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, n=1, shortcut=True, expansion=0.5, act="silu"):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.csam = CSAM(in_channels, in_channels, ksize=ksize, act=act)

        self.conv1 = BaseConv(in_channels, hidden_channels, ksize=ksize, stride=1, act=act)

        self.conv2 = BaseConv(in_channels, hidden_channels, ksize=ksize, stride=1, act=act)

        self.conv3 = BaseConv(2 * hidden_channels, out_channels, ksize=ksize, stride=1, act=act)

        module_list = [Bottleneck(hidden_channels, hidden_channels, shortcut, e=1.0, act=act) for _ in
                       range(n)]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x = self.csam(x)
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)

class CSPDarknet(nn.Module):
    def __init__(self, dep_mul, wid_mul, out_features=("dark3", "dark4", "dark5"), act="silu"):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = BaseConv
        # -----------------------------------------------#
        # The input image is 640, 640, 3
        # The initial base channel is 64
        # -----------------------------------------------#
        base_channels = int(wid_mul * 64)  # 64
        # -----------------------------------------------#
        # Feature extraction using focus network structure
        # 640, 640, 3 -> 320, 320, 12 -> 320, 320, 64
        # -----------------------------------------------#
        self.stem = Focus(3, base_channels, ksize=3, act=act)
        # -----------------------------------------------#
        # After convolution, 320, 320, 64 -> 160, 160, 128
        # After CSPAM, 160, 160, 128 -> 160, 160, 128
        # -----------------------------------------------#
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPAM(base_channels * 2, base_channels * 2, act=act),
        )
        # -----------------------------------------------#
        # After convolution, 160, 160, 128 -> 80, 80, 256
        # After CSPAM, 80, 80, 256 -> 80, 80, 256
        # -----------------------------------------------#
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPAM(base_channels * 4, base_channels * 4, act=act),
        )
        # -----------------------------------------------#
        # After convolution, 80, 80, 256 -> 40, 40, 512
        # After CSPAM, 40, 40, 512 -> 40, 40, 512
        # -----------------------------------------------#
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPAM(base_channels * 8, base_channels * 8, act=act),
        )
        # -----------------------------------------------#
        # After convolution, 40, 40, 512 -> 20, 20, 1024
        # After SFEM, 20, 20, 1024 -> 20, 20, 1024
        # After CSPAM, 20, 20, 1024 -> 20, 20, 1024
        # -----------------------------------------------#
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SFEM(base_channels * 16, base_channels * 16, act=act),
            CSPAM(base_channels * 16, base_channels * 16, shortcut=False, act=act),
        )
    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        # -----------------------------------------------#
        # dark3's output is 80, 80, 256, a valid feature layer
        # -----------------------------------------------#
        x = self.dark3(x)
        outputs["dark3"] = x
        # -----------------------------------------------#
        # dark4's output of 40, 40, 512 is a valid feature layer
        # -----------------------------------------------#
        x = self.dark4(x)
        outputs["dark4"] = x
        # -----------------------------------------------#
        # dark5's output of 20, 20, 1024 is a valid feature layer
        # -----------------------------------------------#
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


if __name__ == '__main__':
    print(CSPDarknet(1, 1))