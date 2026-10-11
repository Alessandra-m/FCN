"""FCN with a skip connection (FCN-16s/8s idea from Long et al., CVPR 2015)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNSkip(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()

        # Encoder: conv3x3 + ReLU + maxpool, stride 2 per block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Pixel classifiers: deep (stride 8) and skip branch (stride 4)
        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)
        self.score_pool2 = nn.Conv2d(64, num_classes, kernel_size=1)

        # Upsampling: x2 to fuse with the skip branch, then x4 to input size
        self.upsample2 = nn.ConvTranspose2d(num_classes, num_classes,
                                            kernel_size=4, stride=2, padding=1, bias=False)
        self.upsample4 = nn.ConvTranspose2d(num_classes, num_classes,
                                            kernel_size=8, stride=4, padding=2, bias=False)
        self.upsample2.weight.data.copy_(bilinear_kernel(num_classes, 4))
        self.upsample4.weight.data.copy_(bilinear_kernel(num_classes, 8))

        # Zero-init the skip branch so training starts from the no-skip predictions
        nn.init.zeros_(self.score_pool2.weight)
        nn.init.zeros_(self.score_pool2.bias)

    def forward(self, x, verbose=False):
        def show(name, t):
            if verbose:
                print(f"  {name:<20} {list(t.shape)}")

        show("input", x)
        H, W = x.shape[2], x.shape[3]

        # pad to a multiple of the total stride so feature maps align and the output matches the input
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        x = F.pad(x, (0, pad_w, 0, pad_h))
        show("padded", x)

        x = self.pool1(F.relu(self.conv1(x))); show("block1 (1/2)", x)
        p2 = self.pool2(F.relu(self.conv2(x))); show("block2 (1/4)", p2)
        p3 = self.pool3(F.relu(self.conv3(p2))); show("block3 (1/8)", p3)

        score8 = self.classifier(p3); show("score (1/8)", score8)
        up = self.upsample2(score8); show("upsample x2", up)
        skip = self.score_pool2(p2); show("score_pool2 (1/4)", skip)
        fused = up + skip; show("fuse (sum)", fused)

        x = self.upsample4(fused); show("upsample x4", x)
        x = x[:, :, :H, :W]; show("output (cropped)", x)
        return x


def bilinear_kernel(channels, kernel_size):
    """ConvTranspose2d weights that perform per-channel bilinear upsampling."""
    factor = (kernel_size + 1) // 2
    center = factor - 0.5 if kernel_size % 2 == 0 else factor - 1
    coords = torch.arange(kernel_size, dtype=torch.float32)
    filt_1d = 1 - torch.abs(coords - center) / factor
    filt_2d = filt_1d[:, None] * filt_1d[None, :]
    weight = torch.zeros(channels, channels, kernel_size, kernel_size)
    for c in range(channels):
        weight[c, c] = filt_2d
    return weight


if __name__ == "__main__":
    torch.manual_seed(0)
    model = FCNSkip(num_classes=21).eval()

    for batch, H, W in [(1, 256, 256), (2, 375, 500)]:
        print(f"\n=== batch={batch}, image {H}x{W} ===")
        img = torch.randn(batch, 3, H, W)
        with torch.no_grad():
            out = model(img, verbose=True)
        mask = out.argmax(dim=1)
        print(f"  {'mask (argmax)':<20} {list(mask.shape)}")
        assert out.shape == (batch, 21, H, W)