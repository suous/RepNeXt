import torch
import torch.nn as nn

from timm.layers import trunc_normal_
from timm.models import register_model, create_model, build_model_with_cfg


class ConvNorm(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        bn_weight_init=1,
    ):
        super().__init__()
        self.add_module("conv", nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias))
        self.add_module("norm", nn.BatchNorm2d(out_channels))
        nn.init.constant_(self.norm.weight, bn_weight_init)
        nn.init.constant_(self.norm.bias, 0)

    @torch.no_grad()
    def fuse(self):
        w = self.norm.weight / (self.norm.running_var + self.norm.eps) ** 0.5
        b = self.norm.bias - w * self.norm.running_mean

        if self.conv.bias is not None:
            b += w * self.conv.bias

        w = w[:, None, None, None] * self.conv.weight

        m = nn.Conv2d(
            w.size(1) * self.conv.groups,
            w.size(0),
            w.shape[2:],
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
            device=self.conv.weight.device,
        )
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class NormLinear(nn.Sequential):
    def __init__(self, in_channels, out_channels, bias=True, std=0.02):
        super().__init__()
        self.add_module("norm", nn.BatchNorm1d(in_channels))
        self.add_module("linear", nn.Linear(in_channels, out_channels, bias=bias))
        trunc_normal_(self.linear.weight, std=std)
        if bias:
            nn.init.constant_(self.linear.bias, 0)

    @torch.no_grad()
    def fuse(self):
        norm, linear = self._modules.values()
        w = norm.weight / (norm.running_var + norm.eps) ** 0.5
        b = norm.bias - self.norm.running_mean * self.norm.weight / (norm.running_var + norm.eps) ** 0.5
        w = linear.weight * w[None, :]
        if linear.bias is None:
            b = b @ self.linear.weight.T
        else:
            b = (linear.weight @ b[:, None]).view(-1) + self.linear.bias
        m = nn.Linear(w.size(1), w.size(0), device=linear.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


def mlp(in_channels, hidden_channels, act_layer=nn.GELU):
    return nn.Sequential(
        ConvNorm(in_channels, hidden_channels, kernel_size=1),
        act_layer(),
        ConvNorm(hidden_channels, in_channels, kernel_size=1),
    )


class RepDWConvS(nn.Module):
    def __init__(self, in_channels, stride=1, bias=True):
        super().__init__()
        self.stride = stride
        kwargs = {"in_channels": in_channels, "out_channels": in_channels, "groups": in_channels}
        self.conv_3_3 = nn.Conv2d(bias=bias, kernel_size=3, stride=stride, dilation=1, padding=1, **kwargs)
        self.conv_3_w = nn.Conv2d(bias=bias and stride==1, kernel_size=(1, 3), stride=(1, stride), padding=(0, 1), **kwargs)
        self.conv_3_h = nn.Conv2d(bias=bias and stride==1, kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0), **kwargs)
        self.conv_2_2 = nn.Conv2d(bias=bias, kernel_size=2, stride=stride, dilation=2, padding=1, **kwargs)

    def forward(self, x):
        if self.stride == 1:
            return self.conv_3_3(x) + self.conv_3_h(x) + self.conv_3_w(x) + self.conv_2_2(x)
        return self.conv_3_3(x) + self.conv_3_h(self.conv_3_w(x)) + self.conv_2_2(x)

    @torch.no_grad()
    def fuse(self):
        conv_3_3_w, conv_3_3_b = self.conv_3_3.weight, self.conv_3_3.bias
        conv_2_2_w, conv_2_2_b = self.conv_2_2.weight, self.conv_2_2.bias
        conv_3_w_w, conv_3_w_b = self.conv_3_w.weight, self.conv_3_w.bias
        conv_3_h_w, conv_3_h_b = self.conv_3_h.weight, self.conv_3_h.bias

        conv_2_2_w = nn.functional.conv_transpose2d(conv_2_2_w, torch.ones((1, 1, 1, 1), device=conv_2_2_w.device), stride=2)
        if self.stride == 2:
            conv_stack_3_w = torch.einsum("bcnx,bcyn->bcyx", conv_3_w_w, conv_3_h_w)
            w = conv_3_3_w + conv_stack_3_w + conv_2_2_w
        else:
            conv_3_w_w = nn.functional.pad(conv_3_w_w, [0, 0, 1, 1])
            conv_3_h_w = nn.functional.pad(conv_3_h_w, [1, 1, 0, 0])
            w = conv_3_3_w + conv_3_w_w + conv_3_h_w + conv_2_2_w
        self.conv_3_3.weight.data.copy_(w)

        if conv_3_3_b is not None:
            b = conv_3_3_b + conv_2_2_b
            if self.stride == 1:
                b += conv_3_w_b + conv_3_h_b
            self.conv_3_3.bias.data.copy_(b)
        return self.conv_3_3


class RepDWConvM(nn.Module):
    def __init__(self, in_channels, stride=1, bias=True):
        super().__init__()
        kwargs = {"in_channels": in_channels, "out_channels": in_channels, "groups": in_channels}
        self.conv_7_7 = nn.Conv2d(bias=bias, kernel_size=(7, 7), stride=stride, padding=3, **kwargs)
        self.conv_5_3 = nn.Conv2d(bias=bias, kernel_size=(5, 3), stride=stride, padding=(2, 1), **kwargs)
        self.conv_3_5 = nn.Conv2d(bias=bias, kernel_size=(3, 5), stride=stride, padding=(1, 2), **kwargs)
        self.conv_7_w = nn.Conv2d(bias=False, kernel_size=(1, 7), stride=(1, stride), padding=(0, 3), **kwargs)
        self.conv_7_h = nn.Conv2d(bias=False, kernel_size=(7, 1), stride=(stride, 1), padding=(3, 0), **kwargs)
        self.conv_5_w = nn.Conv2d(bias=False, kernel_size=(1, 5), stride=(1, stride), padding=(0, 2), **kwargs)
        self.conv_5_h = nn.Conv2d(bias=False, kernel_size=(5, 1), stride=(stride, 1), padding=(2, 0), **kwargs)

    def forward(self, x):
        return self.conv_7_7(x) + self.conv_5_3(x) + self.conv_3_5(x) + self.conv_7_h(self.conv_7_w(x)) + self.conv_5_h(self.conv_5_w(x))

    @torch.no_grad()
    def fuse(self):
        conv_7_7_w, conv_7_7_b = self.conv_7_7.weight, self.conv_7_7.bias
        conv_5_3_w, conv_5_3_b = self.conv_5_3.weight, self.conv_5_3.bias
        conv_3_5_w, conv_3_5_b = self.conv_3_5.weight, self.conv_3_5.bias
        conv_7_w_w, conv_7_h_w = self.conv_7_w.weight, self.conv_7_h.weight
        conv_5_w_w, conv_5_h_w = self.conv_5_w.weight, self.conv_5_h.weight

        conv_5_3_w = nn.functional.pad(conv_5_3_w, [2, 2, 1, 1])
        conv_3_5_w = nn.functional.pad(conv_3_5_w, [1, 1, 2, 2])

        conv_stack_7_w = torch.einsum("bcnx,bcyn->bcyx", conv_7_w_w, conv_7_h_w)
        conv_stack_5_w = torch.einsum("bcnx,bcyn->bcyx", conv_5_w_w, conv_5_h_w)
        conv_stack_5_w = nn.functional.pad(conv_stack_5_w, [1, 1, 1, 1])

        w = conv_7_7_w + conv_5_3_w + conv_3_5_w + conv_stack_7_w + conv_stack_5_w
        self.conv_7_7.weight.data.copy_(w)

        if conv_7_7_b is not None:
            b = conv_7_7_b + conv_5_3_b + conv_3_5_b
            self.conv_7_7.bias.data.copy_(b)
        return self.conv_7_7


class ChunkConv(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        assert in_channels % 4 == 0
        hidden_channels = in_channels // 4
        self.conv_s = RepDWConvS(hidden_channels)
        self.conv_m = RepDWConvM(hidden_channels)
        self.conv_l = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels, 
                out_channels=hidden_channels, 
                kernel_size=(1, 11), 
                padding=(0, 5), 
                groups=hidden_channels
            ),
            nn.Conv2d(
                in_channels=hidden_channels, 
                out_channels=hidden_channels, 
                kernel_size=(11, 1), 
                padding=(5, 0), 
                groups=hidden_channels
            ),
        )

    def forward(self, x):
        i, s, m, l = torch.chunk(x, 4, dim=1)
        return torch.cat((i, self.conv_s(s), self.conv_m(m), self.conv_l(l)), dim=1)


class CopyConv(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_s = RepDWConvS(in_channels, stride=2)
        self.conv_m = RepDWConvM(in_channels, stride=2)

    def forward(self, x):
        return torch.cat((self.conv_s(x), self.conv_m(x)), dim=1)


class RepNextStem(nn.Module):
    def __init__(self, in_channels, out_channels, act_layer=nn.GELU, kernel_size=3, stride=2):
        super().__init__()
        padding = (kernel_size - 1) // 2
        kwargs = {"kernel_size": kernel_size, "stride": stride, "padding": padding}
        self.stem = nn.Sequential(
            ConvNorm(in_channels, out_channels // 2, **kwargs),
            act_layer(),
            ConvNorm(out_channels // 2, out_channels, **kwargs),
        )

    def forward(self, x):
        return self.stem(x)


class MetaNeXtBlock(nn.Module):
    def __init__(self, in_channels, mlp_ratio, act_layer=nn.GELU):
        super().__init__()
        self.token_mixer = ChunkConv(in_channels)
        self.norm = nn.BatchNorm2d(in_channels)
        self.channel_mixer = mlp(in_channels, in_channels * mlp_ratio, act_layer=act_layer)

    def forward(self, x):
        return x + self.channel_mixer(self.norm(self.token_mixer(x)))


class Downsample(nn.Module):
    def __init__(self, in_channels, mlp_ratio, act_layer=nn.GELU):
        super().__init__()
        out_channels = in_channels * 2
        self.token_mixer = CopyConv(in_channels)
        self.norm = nn.BatchNorm2d(out_channels)
        self.channel_mixer = mlp(out_channels, out_channels * mlp_ratio, act_layer=act_layer)

    def forward(self, x):
        x = self.norm(self.token_mixer(x))
        return x + self.channel_mixer(x)


class RepNextClassifier(nn.Module):
    def __init__(self, dim, num_classes, distillation=False, drop=0.0):
        super().__init__()
        self.head_drop = nn.Dropout(drop)
        self.head = NormLinear(dim, num_classes) if num_classes > 0 else nn.Identity()
        self.distillation = distillation
        self.num_classes = num_classes
        self.head_dist = NormLinear(dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        x = self.head_drop(x)
        x1, x2 = self.head(x), self.head_dist(x)
        if self.training and self.distillation and not torch.jit.is_scripting():
            return x1, x2
        else:
            return (x1 + x2) / 2

    @torch.no_grad()
    def fuse(self):
        if not self.num_classes > 0:
            return nn.Identity()
        head = self.head.fuse()
        head_dist = self.head_dist.fuse()
        head.weight += head_dist.weight
        head.bias += head_dist.bias
        head.weight /= 2
        head.bias /= 2
        return head


class RepNextStage(nn.Module):
    def __init__(self, in_channels, out_channels, depth, mlp_ratio, act_layer=nn.GELU, downsample=True):
        super().__init__()
        self.downsample = Downsample(in_channels, mlp_ratio, act_layer=act_layer) if downsample else nn.Identity()
        self.blocks = nn.Sequential(*[MetaNeXtBlock(out_channels, mlp_ratio, act_layer=act_layer) for _ in range(depth)])

    def forward(self, x):
        return self.blocks(self.downsample(x))


class RepNext(nn.Module):
    def __init__(
        self,
        in_chans=3,
        embed_dim=(48,),
        depth=(2,),
        mlp_ratio=2,
        global_pool="avg",
        num_classes=1000,
        act_layer=nn.GELU,
        distillation=True,
        drop_rate=0.0,
    ):
        super().__init__()
        self.global_pool = global_pool
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        in_channels = embed_dim[0]
        self.stem = RepNextStem(in_chans, in_channels, act_layer=act_layer)
        stride = 4
        self.feature_info = []
        stages = []
        for i in range(len(embed_dim)):
            downsample = True if i != 0 else False
            stages.append(
                RepNextStage(
                    in_channels,
                    embed_dim[i],
                    depth[i],
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    downsample=downsample,
                )
            )
            stage_stride = 2 if downsample else 1
            stride *= stage_stride
            self.feature_info += [dict(num_chs=embed_dim[i], reduction=stride, module=f"stages.{i}")]
            in_channels = embed_dim[i]
        self.stages = nn.Sequential(*stages)

        self.num_features = embed_dim[-1]
        self.head_drop = nn.Dropout(drop_rate)
        self.head = RepNextClassifier(embed_dim[-1], num_classes, distillation)

    def forward_features(self, x):
        return self.stages(self.stem(x))

    def forward_head(self, x):
        if self.global_pool == "avg":
            x = x.mean((2, 3), keepdim=False)
        return self.head(self.head_drop(x))

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    @torch.no_grad()
    def fuse(self):
        def fuse_children(net):
            for child_name, child in net.named_children():
                if hasattr(child, "fuse"):
                    fused = child.fuse()
                    setattr(net, child_name, fused)
                    fuse_children(fused)
                else:
                    fuse_children(child)

        fuse_children(self)


def _create_repnext(variant, pretrained=False, **kwargs):
    out_indices = kwargs.pop("out_indices", (0, 1, 2, 3))
    model = build_model_with_cfg(
        RepNext,
        variant,
        pretrained,
        feature_cfg=dict(flatten_sequential=True, out_indices=out_indices),
        **kwargs,
    )
    return model

@register_model
def repnext_m0(pretrained=False, **kwargs):
    model_args = dict(embed_dim=(40, 80, 160, 320), depth=(2, 2, 9, 1))
    return _create_repnext("repnext_m0", pretrained=pretrained, **dict(model_args, **kwargs))

@register_model
def repnext_m1(pretrained=False, **kwargs):
    model_args = dict(embed_dim=(48, 96, 192, 384), depth=(3, 3, 15, 2))
    return _create_repnext("repnext_m1", pretrained=pretrained, **dict(model_args, **kwargs))

@register_model
def repnext_m2(pretrained=False, **kwargs):
    model_args = dict(embed_dim=(56, 112, 224, 448), depth=(3, 3, 15, 2))
    return _create_repnext("repnext_m2", pretrained=pretrained, **dict(model_args, **kwargs))

@register_model
def repnext_m3(pretrained=False, **kwargs):
    model_args = dict(embed_dim=(64, 128, 256, 512), depth=(3, 3, 13, 2))
    return _create_repnext("repnext_m3", pretrained=pretrained, **dict(model_args, **kwargs))

@register_model
def repnext_m4(pretrained=False, **kwargs):
    model_args = dict(embed_dim=(64, 128, 256, 512), depth=(5, 5, 25, 4))
    return _create_repnext("repnext_m4", pretrained=pretrained, **dict(model_args, **kwargs))

@register_model
def repnext_m5(pretrained=False, **kwargs):
    model_args = dict(embed_dim=(80, 160, 320, 640), depth=(7, 7, 35, 2))
    return _create_repnext("repnext_m5", pretrained=pretrained, **dict(model_args, **kwargs))


if __name__ == "__main__":
    model = create_model("repnext_m3")

    model.eval()
    print(str(model))
    try:
        import pytorch_model_summary
        x = torch.randn(1, 3, 224, 224)
        print(pytorch_model_summary.summary(model, x))
    except ModuleNotFoundError:
        pass

