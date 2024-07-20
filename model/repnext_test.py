import torch
from torch import nn

import itertools
import unittest
from repnext import RepDWConvM, RepDWConvS, RepNextClassifier


class ChunkConv(nn.Module):
    def __init__(self, in_channels, bias=True):
        super().__init__()
        self.bias = bias
        in_channels = in_channels // 4
        kwargs = {"in_channels": in_channels, "out_channels": in_channels, "groups": in_channels, "bias": bias}
        self.conv_i = nn.Identity()
        self.conv_s = nn.Conv2d(kernel_size=3, padding=1, **kwargs)
        self.conv_m = nn.Conv2d(kernel_size=7, padding=3, **kwargs)
        self.conv_l = nn.Conv2d(kernel_size=11, padding=5, **kwargs)
         
    def forward(self, x):
        i, s, m, l = torch.chunk(x, chunks=4, dim=1)
        return torch.cat((self.conv_i(i), self.conv_s(s), self.conv_m(m), self.conv_l(l)), dim=1)

    @torch.no_grad()
    def fuse(self):
        conv_s_w, conv_s_b = self.conv_s.weight, self.conv_s.bias
        conv_m_w, conv_m_b = self.conv_m.weight, self.conv_m.bias
        conv_l_w, conv_l_b = self.conv_l.weight, self.conv_l.bias

        conv_i_w = torch.nn.functional.pad(torch.ones(conv_l_w.shape[0], conv_l_w.shape[1], 1, 1), [5, 5, 5, 5])
        conv_s_w = nn.functional.pad(conv_s_w, [4, 4, 4, 4])
        conv_m_w = nn.functional.pad(conv_m_w, [2, 2, 2, 2])

        in_channels = self.conv_l.in_channels*4
        conv = nn.Conv2d(in_channels, in_channels, kernel_size=11, padding=5, bias=self.bias, groups=in_channels)
        conv.weight.data.copy_(torch.cat((conv_i_w, conv_s_w, conv_m_w, conv_l_w), dim=0))

        if self.bias:
            conv_i_b = torch.zeros_like(conv_s_b)
            conv.bias.data.copy_(torch.cat((conv_i_b, conv_s_b, conv_m_b, conv_l_b), dim=0))
        return conv


class CopyConv(nn.Module):
    def __init__(self, in_channels, bias=True):
        super().__init__()
        self.bias = bias
        kwargs = {"in_channels": in_channels, "out_channels": in_channels, "groups": in_channels, "bias": bias, "stride": 2}
        self.conv_s = nn.Conv2d(kernel_size=3, padding=1, **kwargs)
        self.conv_l = nn.Conv2d(kernel_size=7, padding=3, **kwargs)
         
    def forward(self, x):
        B, C, H, W = x.shape
        s, l = self.conv_s(x), self.conv_l(x)
        return torch.stack((s, l), dim=2).reshape(B, C*2, H//2, W//2)

    @torch.no_grad()
    def fuse(self):
        conv_s_w, conv_s_b = self.conv_s.weight, self.conv_s.bias
        conv_l_w, conv_l_b = self.conv_l.weight, self.conv_l.bias

        conv_s_w = nn.functional.pad(conv_s_w, [2, 2, 2, 2])

        in_channels = self.conv_l.in_channels
        conv = nn.Conv2d(in_channels, in_channels*2, kernel_size=7, padding=3, bias=self.bias, stride=self.conv_l.stride, groups=in_channels)
        conv.weight.data.copy_(torch.stack((conv_s_w, conv_l_w), dim=1).reshape(conv.weight.shape))

        if self.bias:
            conv.bias.data.copy_(torch.stack((conv_s_b, conv_l_b), dim=1).reshape(conv.bias.shape))
        return conv


class TestRepCases(unittest.TestCase):
    batch_size = 32
    in_channels = 64
    img_size = 56

    def test_rep_dw_conv_m(self):
        x = torch.randn(self.batch_size, self.in_channels, self.img_size, self.img_size)
        for stride, bias in itertools.product([1, 2], [False, True]):
            with self.subTest(stride=stride, bias=bias):
                cn = RepDWConvM(in_channels=self.in_channels, stride=stride, bias=bias)
                cn.eval()
                self.assertTrue(torch.allclose(cn(x), cn.fuse()(x), atol=5e-5), "RepDWConvM failed")

    def test_rep_dw_conv_s(self):
        x = torch.randn(self.batch_size, self.in_channels, self.img_size, self.img_size)
        for stride, bias in itertools.product([1, 2], [False, True]):
            with self.subTest(stride=stride, bias=bias):
                cn = RepDWConvS(in_channels=self.in_channels, stride=stride, bias=bias)
                cn.eval()
                self.assertTrue(torch.allclose(cn(x), cn.fuse()(x), atol=5e-5), "RepDWConvS failed")

    def test_rep_next_classifier(self):
        x = torch.randn(self.batch_size, self.in_channels, self.img_size, self.img_size)
        x = x.mean((2, 3), keepdim=False)
        for classes, distillation in itertools.product([100, 1000], [False, True]):
            with self.subTest(classes=classes, distillation=distillation):
                cn = RepNextClassifier(self.in_channels, classes, distillation)
                cn.eval()
                self.assertTrue(torch.allclose(cn(x), cn.fuse()(x), atol=5e-5), "RepNextClassifier failed")

    def test_chunk_conv(self):
        x = torch.randn(self.batch_size, self.in_channels, self.img_size, self.img_size)
        for bias in [False, True]:
            with self.subTest(bias=bias):
                cn = ChunkConv(in_channels=self.in_channels, bias=bias)
                cn.eval()
                self.assertTrue(torch.allclose(cn(x), cn.fuse()(x), atol=5e-5), "ChunkConv failed")

    def test_copy_conv(self):
        x = torch.randn(self.batch_size, self.in_channels, self.img_size, self.img_size)
        for bias in [False, True]:
            with self.subTest(bias=bias):
                cn = CopyConv(in_channels=self.in_channels, bias=bias)
                cn.eval()
                self.assertTrue(torch.allclose(cn(x), cn.fuse()(x), atol=5e-5), "CopyConv failed")


if __name__ == '__main__':
    unittest.main()
