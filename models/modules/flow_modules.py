from torch import nn


class FlowBlock(nn.Module):
    def __init__(self, input_nc, ngf=64, norm_layer=nn.BatchNorm2d):
        super(FlowBlock, self).__init__()

        # FIXME: 가장 마지막 layer로 normalization layer가 사용되는 특별한 이유가 있을까요?
        self.model = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1), nn.ReLU(True), norm_layer(ngf),
            nn.Conv2d(ngf, ngf, kernel_size=3, stride=2, padding=1), nn.ReLU(True), norm_layer(ngf),
            nn.Conv2d(ngf, ngf, kernel_size=3, stride=2, padding=1), nn.ReLU(True), norm_layer(ngf),
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1), nn.ReLU(True), norm_layer(ngf)
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
