import torch
import itertools
import unittest
from repnext import RepDWConvM, RepDWConvS, RepNextClassifier


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


if __name__ == '__main__':
    unittest.main()
