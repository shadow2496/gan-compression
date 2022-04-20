from torch import nn


class FlowBlock(nn.Module):
    def __init__(self, input_nc, ngf=64, norm_layer=nn.InstanceNorm2d):
        super(FlowBlock, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(input_nc, input_nc, kernel_size=3, padding=1, groups=input_nc), norm_layer(input_nc, eps=1e-04), nn.Conv2d(input_nc, ngf, kernel_size=1),
            nn.ReLU(True), norm_layer(ngf, eps=1e-04),
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, groups=ngf), norm_layer(ngf, eps=1e-04), nn.Conv2d(ngf, ngf, kernel_size=1),
            nn.ReLU(True), norm_layer(ngf, eps=1e-04),
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, groups=ngf), norm_layer(ngf, eps=1e-04), nn.Conv2d(ngf, ngf, kernel_size=1),
            nn.ReLU(True), norm_layer(ngf, eps=1e-04),
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, groups=ngf), norm_layer(ngf, eps=1e-04), nn.Conv2d(ngf, ngf, kernel_size=1),
            nn.ReLU(True), norm_layer(ngf, eps=1e-04),
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, groups=ngf), norm_layer(ngf, eps=1e-04), nn.Conv2d(ngf, ngf, kernel_size=1),
        )

    def forward(self, x):
        return self.model(x)


class FlowNet(nn.Module):
    def __init__(self, input_nc, ngfs):
        super(FlowNet, self).__init__()

        layers = []
        for ngf in ngfs:
            layers += [FlowBlock(input_nc, ngf)]

        self.model = nn.Sequential(*layers)

    def forward(self, x, idx):
        return self.model[idx](x)
