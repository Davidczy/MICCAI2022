import timm
import torch
import torch.nn as nn
from itertools import repeat
import collections.abc
from timm.models import create_model

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

class PatchEmbed2D(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class PatchEmbed3D(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=(128,112,112), patch_size=(8,8,8), in_chans=1, embed_dim=96, norm_layer=None, flatten=True):
        super().__init__()
        # img_size = to_3tuple(img_size)
        # patch_size = to_3tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.flatten = flatten

        self.proj1 = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W, D = x.shape #增加深度信息
        assert H == self.img_size[0] and W == self.img_size[1] and D == self.img_size[2], \
            f"Input image size ({H}*{W}*{D}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        x = self.proj1(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHWD -> BNC
        x = self.norm(x)
        return x


class UniformerS1(nn.Module):
    def __init__(self):
        super(UniformerS1, self).__init__()
        self.twoDemd = timm.create_model('swin_tiny_patch4_window7_224', num_classes=0, pretrained=True)
        self.threeDemd = timm.create_model('swin_tiny_patch4_window7_224', num_classes=0, pretrained=True)
        self.threeDemd.patch_embed = PatchEmbed3D()

    def forward(self, input):
        if len(input.shape) == 4:
            x = self.twoDemd(input)
        else:
            x = self.threeDemd(input)
        return x