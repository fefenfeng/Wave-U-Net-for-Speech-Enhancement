import torch
import torch.nn as nn
import torch.nn.functional as F


class DownSamplingLayer(nn.Module):

    def __init__(self, channel_in, channel_out, dilation=1, kernel_size=15, stride=1, padding=7):
        """
        Args:
            channel_in:
            channel_out:
            dilation: 默认为1，不空洞卷积，2为空洞卷积
            kernel_size:
            stride:步长默认为1
            padding: 步长为1时，为了让序列长度一直,padding = (kernel_size - 1) / 2
        Notes:
            下采样块，较粗时间尺度上计算特征，包含conv1d，batchnorm，leakyReLu
        """
        super(DownSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, ipt):
        return self.main(ipt)


class UpSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=5, stride=1, padding=2):
        super(UpSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            # inplace是否选择覆盖运算，节省运算，直接对原值进行操作
            # 但可能会覆盖输入数据，无法求取relu梯度，训练中可能梯度爆炸loss飙升
        )

    def forward(self, ipt):
        return self.main(ipt)


class Model(nn.Module):
    def __init__(self, n_layers=12, channels_interval=24):
        super(Model, self).__init__()

        self.n_layers = n_layers
        self.channels_interval = channels_interval
        # 通道数: 刚开始单通道, 1 -> 24 -> 24 * 2 -> 24 * 11，其中用了12层
        encoder_in_channels_list = [1] + [i * self.channels_interval for i in range(1, self.n_layers)]
        # [ 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288 ]
        encoder_out_channels_list = [i * self.channels_interval for i in range(1, self.n_layers + 1)]

        self.encoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.encoder.append(
                DownSamplingLayer(
                    channel_in=encoder_in_channels_list[i],
                    channel_out=encoder_out_channels_list[i]
                )
            )

        # 中间层，对通道数不做改变，通道数为 12 * 24在这里
        self.middle = nn.Sequential(
            nn.Conv1d(self.n_layers * self.channels_interval, self.n_layers * self.channels_interval, 15, stride=1,
                      padding=7),
            nn.BatchNorm1d(self.n_layers * self.channels_interval),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        # [ 72, 120, 168, 216, 264, 312, 360, 408, 456, 504, 552, 576 ]
        decoder_in_channels_list = [(2 * i + 1) * self.channels_interval for i in range(1, self.n_layers)] + [
            2 * self.n_layers * self.channels_interval]
        # [576, 552, 504, 456, 408, 360, 312, 264, 216, 168, 120, 72]
        decoder_in_channels_list = decoder_in_channels_list[::-1]
        # [288, 264, 240, 216, 192, 168, 144, 120, 96, 72, 48, 24]
        decoder_out_channels_list = encoder_out_channels_list[::-1]

        self.decoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.decoder.append(
                UpSamplingLayer(
                    channel_in=decoder_in_channels_list[i],
                    channel_out=decoder_out_channels_list[i]
                )
            )

        self.out = nn.Sequential(
            nn.Conv1d(1 + self.channels_interval, 1, kernel_size=1, stride=1),
            nn.Tanh()
        )

    def forward(self, input):
        tmp = []
        o = input

        # Up Sampling
        # 进encoder，一直对时间序列长度 // 2, 最后一个为(batch_size, T = 4, channel = 288)
        for i in range(self.n_layers):
            o = self.encoder[i](o)
            tmp.append(o)
            # [batch_size, T // 2, channels]
            o = o[:, :, ::2]

        o = self.middle(o)  # 最后一个encoder的过middle

        # Down Sampling
        for i in range(self.n_layers):
            # [batch_size, T * 2, channels]
            o = F.interpolate(o, scale_factor=2, mode="linear", align_corners=True)
            # Skip Connection
            o = torch.cat([o, tmp[self.n_layers - i - 1]], dim=1)
            o = self.decoder[i](o)

        o = torch.cat([o, input], dim=1)
        o = self.out(o)
        return o
