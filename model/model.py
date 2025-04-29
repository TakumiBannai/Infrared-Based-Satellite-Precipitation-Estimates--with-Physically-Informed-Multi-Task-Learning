import torch
import torch.nn as nn


class ConvBNRelu(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                              kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class PoolingBNRelu(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.in_channel = in_channel
        self.maxpooling = nn.MaxPool2d(2, 2, 0)
        self.bn = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.maxpooling(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvReluConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel*2,
                               kernel_size=3, stride=1, padding=1, dilation=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=in_channel*2, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, dilation=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class DeconvConvConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.deconv = nn.ConvTranspose2d(in_channels=in_channel,
                                         out_channels=in_channel,
                                         kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel,
                               kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, dilation=1)

    def forward(self, x):
        x = self.deconv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# Single-task learning model
class STL_IR(nn.Module):
    def __init__(self):
        super().__init__()
        # IR channel encoders
        self.conv_IR_ch08 = nn.Sequential(ConvBNRelu(2, 16),
                                          PoolingBNRelu(16),
                                          ConvBNRelu(16, 32),
                                          PoolingBNRelu(32)
                                          )
        self.conv_IR_ch10 = nn.Sequential(ConvBNRelu(2, 16),
                                          PoolingBNRelu(16),
                                          ConvBNRelu(16, 32),
                                          PoolingBNRelu(32)
                                          )
        self.conv_IR_ch11 = nn.Sequential(ConvBNRelu(2, 16),
                                          PoolingBNRelu(16),
                                          ConvBNRelu(16, 32),
                                          PoolingBNRelu(32)
                                          )
        self.conv_IR_ch14 = nn.Sequential(ConvBNRelu(2, 16),
                                          PoolingBNRelu(16),
                                          ConvBNRelu(16, 32),
                                          PoolingBNRelu(32)
                                          )
        self.conv_IR_ch15 = nn.Sequential(ConvBNRelu(2, 16),
                                          PoolingBNRelu(16),
                                          ConvBNRelu(16, 32),
                                          PoolingBNRelu(32)
                                          )

        # Decoder network (32*5ch = 160 dim)
        self.decoder = nn.Sequential(DeconvConvConv(160, 64),
                                     DeconvConvConv(64, 128),
                                     nn.Conv2d(128, 128, 3, 1, 1),
                                     nn.ReLU(),
                                     nn.Conv2d(128, 1, 3, 1, 1),
                                     nn.ReLU()
                                     )

    def forward(self, x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15):
        x_ir_ch08 = self.conv_IR_ch08(x_ir_ch08)
        x_ir_ch10 = self.conv_IR_ch10(x_ir_ch10)
        x_ir_ch11 = self.conv_IR_ch11(x_ir_ch11)
        x_ir_ch14 = self.conv_IR_ch14(x_ir_ch14)
        x_ir_ch15 = self.conv_IR_ch15(x_ir_ch15)
        x = torch.cat((x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15), dim=1)
        x = self.decoder(x)
        return x


class STL_IR_CW(nn.Module):
    def __init__(self):
        super().__init__()
        # IR&CW channel encoder
        self.conv_IR_ch08 = nn.Sequential(ConvBNRelu(2, 16),
                                          PoolingBNRelu(16),
                                          ConvBNRelu(16, 32),
                                          PoolingBNRelu(32))
        self.conv_IR_ch10 = nn.Sequential(ConvBNRelu(2, 16),
                                          PoolingBNRelu(16),
                                          ConvBNRelu(16, 32),
                                          PoolingBNRelu(32))
        self.conv_IR_ch11 = nn.Sequential(ConvBNRelu(2, 16),
                                          PoolingBNRelu(16),
                                          ConvBNRelu(16, 32),
                                          PoolingBNRelu(32))
        self.conv_IR_ch14 = nn.Sequential(ConvBNRelu(2, 16),
                                          PoolingBNRelu(16),
                                          ConvBNRelu(16, 32),
                                          PoolingBNRelu(32))
        self.conv_IR_ch15 = nn.Sequential(ConvBNRelu(2, 16),
                                          PoolingBNRelu(16),
                                          ConvBNRelu(16, 32),
                                          PoolingBNRelu(32))
        self.conv_CW = nn.Sequential(ConvBNRelu(2, 16),
                                     PoolingBNRelu(16),
                                     ConvBNRelu(16, 32),
                                     PoolingBNRelu(32))

        # Decoder network (32*6ch = 192 dim)
        self.decoder = nn.Sequential(DeconvConvConv(192, 64),
                                     DeconvConvConv(64, 128),
                                     nn.Conv2d(128, 128, 3, 1, 1),
                                     nn.ReLU(),
                                     nn.Conv2d(128, 1, 3, 1, 1),
                                     nn.ReLU())

    def forward(self, x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw):
        x_ir_ch08 = self.conv_IR_ch08(x_ir_ch08)
        x_ir_ch10 = self.conv_IR_ch10(x_ir_ch10)
        x_ir_ch11 = self.conv_IR_ch11(x_ir_ch11)
        x_ir_ch14 = self.conv_IR_ch14(x_ir_ch14)
        x_ir_ch15 = self.conv_IR_ch15(x_ir_ch15)
        x_cw = self.conv_CW(x_cw)
        x = torch.cat((x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw), dim=1)
        x = self.decoder(x)
        return x


class STL_IR_CI(nn.Module):
    def __init__(self):
        super().__init__()
        # IR&CI channel encoder
        self.conv_IR_ch08 = nn.Sequential(ConvBNRelu(2, 16),
                                          PoolingBNRelu(16),
                                          ConvBNRelu(16, 32),
                                          PoolingBNRelu(32))
        self.conv_IR_ch10 = nn.Sequential(ConvBNRelu(2, 16),
                                          PoolingBNRelu(16),
                                          ConvBNRelu(16, 32),
                                          PoolingBNRelu(32))
        self.conv_IR_ch11 = nn.Sequential(ConvBNRelu(2, 16),
                                          PoolingBNRelu(16),
                                          ConvBNRelu(16, 32),
                                          PoolingBNRelu(32))
        self.conv_IR_ch14 = nn.Sequential(ConvBNRelu(2, 16),
                                          PoolingBNRelu(16),
                                          ConvBNRelu(16, 32),
                                          PoolingBNRelu(32))
        self.conv_IR_ch15 = nn.Sequential(ConvBNRelu(2, 16),
                                          PoolingBNRelu(16),
                                          ConvBNRelu(16, 32),
                                          PoolingBNRelu(32))
        self.conv_CI = nn.Sequential(ConvBNRelu(2, 16),
                                     PoolingBNRelu(16),
                                     ConvBNRelu(16, 32),
                                     PoolingBNRelu(32))

        # Decoder network (32*6ch = 192 dim)
        self.decoder = nn.Sequential(DeconvConvConv(192, 64),
                                     DeconvConvConv(64, 128),
                                     nn.Conv2d(128, 128, 3, 1, 1),
                                     nn.ReLU(),
                                     nn.Conv2d(128, 1, 3, 1, 1),
                                     nn.ReLU())

    def forward(self, x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_ci):
        x_ir_ch08 = self.conv_IR_ch08(x_ir_ch08)
        x_ir_ch10 = self.conv_IR_ch10(x_ir_ch10)
        x_ir_ch11 = self.conv_IR_ch11(x_ir_ch11)
        x_ir_ch14 = self.conv_IR_ch14(x_ir_ch14)
        x_ir_ch15 = self.conv_IR_ch15(x_ir_ch15)
        x_ci = self.conv_CI(x_ci)
        x = torch.cat((x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_ci), dim=1)
        x = self.decoder(x)
        return x


class STL_IR_CWCI(nn.Module):
    def __init__(self):
        super().__init__()
        # IR&CW&CI channel encoders
        self.conv_IR_ch08 = nn.Sequential(ConvBNRelu(2, 16),
                                          PoolingBNRelu(16),
                                          ConvBNRelu(16, 32),
                                          PoolingBNRelu(32))
        self.conv_IR_ch10 = nn.Sequential(ConvBNRelu(2, 16),
                                          PoolingBNRelu(16),
                                          ConvBNRelu(16, 32),
                                          PoolingBNRelu(32))
        self.conv_IR_ch11 = nn.Sequential(ConvBNRelu(2, 16),
                                          PoolingBNRelu(16),
                                          ConvBNRelu(16, 32),
                                          PoolingBNRelu(32))
        self.conv_IR_ch14 = nn.Sequential(ConvBNRelu(2, 16),
                                          PoolingBNRelu(16),
                                          ConvBNRelu(16, 32),
                                          PoolingBNRelu(32))
        self.conv_IR_ch15 = nn.Sequential(ConvBNRelu(2, 16),
                                          PoolingBNRelu(16),
                                          ConvBNRelu(16, 32),
                                          PoolingBNRelu(32))
        self.conv_CW = nn.Sequential(ConvBNRelu(2, 16),
                                     PoolingBNRelu(16),
                                     ConvBNRelu(16, 32),
                                     PoolingBNRelu(32))
        self.conv_CI = nn.Sequential(ConvBNRelu(2, 16),
                                     PoolingBNRelu(16),
                                     ConvBNRelu(16, 32),
                                     PoolingBNRelu(32))

        # Decoder network (32*7ch = 224 dim)
        self.decoder = nn.Sequential(DeconvConvConv(224, 64),
                                     DeconvConvConv(64, 128),
                                     nn.Conv2d(128, 128, 3, 1, 1),
                                     nn.ReLU(),
                                     nn.Conv2d(128, 1, 3, 1, 1),
                                     nn.ReLU())

    def forward(self, x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw, x_ci):
        x_ir_ch08 = self.conv_IR_ch08(x_ir_ch08)
        x_ir_ch10 = self.conv_IR_ch10(x_ir_ch10)
        x_ir_ch11 = self.conv_IR_ch11(x_ir_ch11)
        x_ir_ch14 = self.conv_IR_ch14(x_ir_ch14)
        x_ir_ch15 = self.conv_IR_ch15(x_ir_ch15)
        x_cw = self.conv_CW(x_cw)
        x_ci = self.conv_CI(x_ci)
        x = torch.cat((x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw, x_ci), dim=1)
        x = self.decoder(x)
        return x


# Multi-task learning model
class MTL_CW(nn.Module):
    def __init__(self):
        super().__init__()
        # IR&CW&CI channel encoders
        self.conv_IR_ch08 = nn.Sequential(ConvBNRelu(2, 16),
                                          PoolingBNRelu(16),
                                          ConvBNRelu(16, 32),
                                          PoolingBNRelu(32))
        self.conv_IR_ch10 = nn.Sequential(ConvBNRelu(2, 16),
                                          PoolingBNRelu(16),
                                          ConvBNRelu(16, 32),
                                          PoolingBNRelu(32))
        self.conv_IR_ch11 = nn.Sequential(ConvBNRelu(2, 16),
                                          PoolingBNRelu(16),
                                          ConvBNRelu(16, 32),
                                          PoolingBNRelu(32))
        self.conv_IR_ch14 = nn.Sequential(ConvBNRelu(2, 16),
                                          PoolingBNRelu(16),
                                          ConvBNRelu(16, 32),
                                          PoolingBNRelu(32))
        self.conv_IR_ch15 = nn.Sequential(ConvBNRelu(2, 16),
                                          PoolingBNRelu(16),
                                          ConvBNRelu(16, 32),
                                          PoolingBNRelu(32))
        self.conv_CW = nn.Sequential(ConvBNRelu(2, 16),
                                     PoolingBNRelu(16),
                                     ConvBNRelu(16, 32),
                                     PoolingBNRelu(32))

        # Decoder network (32*6 = 192 ch)
        self.decoder_RR = nn.Sequential(DeconvConvConv(192, 64),
                                        DeconvConvConv(64, 128),
                                        nn.Conv2d(128, 128, 3, 1, 1),
                                        nn.ReLU()
                                        )

        self.decoder_RM = nn.Sequential(DeconvConvConv(192, 64),
                                        DeconvConvConv(64, 128),
                                        nn.Conv2d(128, 128, 3, 1, 1),
                                        nn.ReLU()
                                        )

        self.decoder_CW = nn.Sequential(DeconvConvConv(192, 64),
                                        DeconvConvConv(64, 128),
                                        nn.Conv2d(128, 128, 3, 1, 1),
                                        nn.ReLU()
                                        )

        # Output layer
        self.out_RR = nn.Sequential(ConvReluConv(128, 1),
                                    nn.ReLU()
                                    )

        self.out_RM = nn.Sequential(ConvReluConv(128, 1),
                                    nn.Sigmoid()
                                    )
        self.out_CW = nn.Sequential(ConvReluConv(128, 1),
                                    nn.ReLU()
                                    )

    def forward(self, x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw):
        x_ir_ch08 = self.conv_IR_ch08(x_ir_ch08)
        x_ir_ch10 = self.conv_IR_ch10(x_ir_ch10)
        x_ir_ch11 = self.conv_IR_ch11(x_ir_ch11)
        x_ir_ch14 = self.conv_IR_ch14(x_ir_ch14)
        x_ir_ch15 = self.conv_IR_ch15(x_ir_ch15)
        x_cw = self.conv_CW(x_cw)
        x = torch.cat((x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw), dim=1)
        # Decoder
        x_rainrate = self.decoder_RR(x)
        x_rainmask = self.decoder_RM(x)
        x_cw = self.decoder_CW(x)
        # Output
        x_rainrate = self.out_RR(x_rainrate)
        x_rainmask = self.out_RM(x_rainmask)
        x_cw = self.out_CW(x_cw)
        return x_rainrate, x_rainmask, x_cw


class MTL_CW_multi(nn.Module):
    def __init__(self):
        super().__init__()
        # IR&CW&CI channel encoders
        self.conv_IR_ch08 = nn.Sequential(ConvBNRelu(2, 16),
                                          PoolingBNRelu(16),
                                          ConvBNRelu(16, 32),
                                          PoolingBNRelu(32))
        self.conv_IR_ch10 = nn.Sequential(ConvBNRelu(2, 16),
                                          PoolingBNRelu(16),
                                          ConvBNRelu(16, 32),
                                          PoolingBNRelu(32))
        self.conv_IR_ch11 = nn.Sequential(ConvBNRelu(2, 16),
                                          PoolingBNRelu(16),
                                          ConvBNRelu(16, 32),
                                          PoolingBNRelu(32))
        self.conv_IR_ch14 = nn.Sequential(ConvBNRelu(2, 16),
                                          PoolingBNRelu(16),
                                          ConvBNRelu(16, 32),
                                          PoolingBNRelu(32))
        self.conv_IR_ch15 = nn.Sequential(ConvBNRelu(2, 16),
                                          PoolingBNRelu(16),
                                          ConvBNRelu(16, 32),
                                          PoolingBNRelu(32))
        self.conv_CW = nn.Sequential(ConvBNRelu(2, 16),
                                     PoolingBNRelu(16),
                                     ConvBNRelu(16, 32),
                                     PoolingBNRelu(32))

        # Decoder network (32*6 = 192 ch)
        self.decoder_RR = nn.Sequential(DeconvConvConv(192, 64),
                                        DeconvConvConv(64, 128),
                                        nn.Conv2d(128, 128, 3, 1, 1),
                                        nn.ReLU()
                                        )

        self.decoder_RM = nn.Sequential(DeconvConvConv(192, 64),
                                        DeconvConvConv(64, 128),
                                        nn.Conv2d(128, 128, 3, 1, 1),
                                        nn.ReLU()
                                        )

        self.decoder_CW = nn.Sequential(DeconvConvConv(192, 64),
                                        DeconvConvConv(64, 128),
                                        nn.Conv2d(128, 128, 3, 1, 1),
                                        nn.ReLU()
                                        )

        # Output layer
        self.out_RR = nn.Sequential(ConvReluConv(128, 1),
                                    nn.ReLU()
                                    )

        self.out_RM = nn.Sequential(ConvReluConv(128, 4),
                                    # nn.Sigmoid()  # Multi-class では不要
                                    )
        self.out_CW = nn.Sequential(ConvReluConv(128, 1),
                                    nn.ReLU()
                                    )

    def forward(self, x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw):
        x_ir_ch08 = self.conv_IR_ch08(x_ir_ch08)
        x_ir_ch10 = self.conv_IR_ch10(x_ir_ch10)
        x_ir_ch11 = self.conv_IR_ch11(x_ir_ch11)
        x_ir_ch14 = self.conv_IR_ch14(x_ir_ch14)
        x_ir_ch15 = self.conv_IR_ch15(x_ir_ch15)
        x_cw = self.conv_CW(x_cw)
        x = torch.cat((x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw), dim=1)
        # Decoder
        x_rainrate = self.decoder_RR(x)
        x_rainmask = self.decoder_RM(x)
        x_cw = self.decoder_CW(x)
        # Output
        x_rainrate = self.out_RR(x_rainrate)
        x_rainmask = self.out_RM(x_rainmask)  # no, weak, moderate, heavy
        x_cw = self.out_CW(x_cw)
        return x_rainrate, x_rainmask, x_cw


class MTL_CI(nn.Module):
    def __init__(self):
        super().__init__()
        # IR&CW&CI channel encoders
        self.conv_IR_ch08 = nn.Sequential(ConvBNRelu(2, 16),
                                          PoolingBNRelu(16),
                                          ConvBNRelu(16, 32),
                                          PoolingBNRelu(32))
        self.conv_IR_ch10 = nn.Sequential(ConvBNRelu(2, 16),
                                          PoolingBNRelu(16),
                                          ConvBNRelu(16, 32),
                                          PoolingBNRelu(32))
        self.conv_IR_ch11 = nn.Sequential(ConvBNRelu(2, 16),
                                          PoolingBNRelu(16),
                                          ConvBNRelu(16, 32),
                                          PoolingBNRelu(32))
        self.conv_IR_ch14 = nn.Sequential(ConvBNRelu(2, 16),
                                          PoolingBNRelu(16),
                                          ConvBNRelu(16, 32),
                                          PoolingBNRelu(32))
        self.conv_IR_ch15 = nn.Sequential(ConvBNRelu(2, 16),
                                          PoolingBNRelu(16),
                                          ConvBNRelu(16, 32),
                                          PoolingBNRelu(32))
        self.conv_CI = nn.Sequential(ConvBNRelu(2, 16),
                                     PoolingBNRelu(16),
                                     ConvBNRelu(16, 32),
                                     PoolingBNRelu(32))

        # Decoder network (32*6 = 192 ch)
        self.decoder_RR = nn.Sequential(DeconvConvConv(192, 64),
                                        DeconvConvConv(64, 128),
                                        nn.Conv2d(128, 128, 3, 1, 1),
                                        nn.ReLU()
                                        )

        self.decoder_RM = nn.Sequential(DeconvConvConv(192, 64),
                                        DeconvConvConv(64, 128),
                                        nn.Conv2d(128, 128, 3, 1, 1),
                                        nn.ReLU()
                                        )

        self.decoder_CI = nn.Sequential(DeconvConvConv(192, 64),
                                        DeconvConvConv(64, 128),
                                        nn.Conv2d(128, 128, 3, 1, 1),
                                        nn.ReLU()
                                        )

        # Output layer
        self.out_RR = nn.Sequential(ConvReluConv(128, 1),
                                    nn.ReLU()
                                    )

        self.out_RM = nn.Sequential(ConvReluConv(128, 1),
                                    nn.Sigmoid()
                                    )
        self.out_CI = nn.Sequential(ConvReluConv(128, 1),
                                    nn.ReLU()
                                    )

    def forward(self, x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_ci):
        x_ir_ch08 = self.conv_IR_ch08(x_ir_ch08)
        x_ir_ch10 = self.conv_IR_ch10(x_ir_ch10)
        x_ir_ch11 = self.conv_IR_ch11(x_ir_ch11)
        x_ir_ch14 = self.conv_IR_ch14(x_ir_ch14)
        x_ir_ch15 = self.conv_IR_ch15(x_ir_ch15)
        x_ci = self.conv_CI(x_ci)
        x = torch.cat((x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_ci), dim=1)
        # Decoder
        x_rainrate = self.decoder_RR(x)
        x_rainmask = self.decoder_RM(x)
        x_ci = self.decoder_CI(x)
        # Output
        x_rainrate = self.out_RR(x_rainrate)
        x_rainmask = self.out_RM(x_rainmask)
        x_ci = self.out_CI(x_ci)
        return x_rainrate, x_rainmask, x_ci


class MTL_CWCI(nn.Module):
    def __init__(self):
        super().__init__()
        # IR&CW&CI channel encoders
        self.conv_IR_ch08 = nn.Sequential(ConvBNRelu(2, 16),
                                          PoolingBNRelu(16),
                                          ConvBNRelu(16, 32),
                                          PoolingBNRelu(32))
        self.conv_IR_ch10 = nn.Sequential(ConvBNRelu(2, 16),
                                          PoolingBNRelu(16),
                                          ConvBNRelu(16, 32),
                                          PoolingBNRelu(32))
        self.conv_IR_ch11 = nn.Sequential(ConvBNRelu(2, 16),
                                          PoolingBNRelu(16),
                                          ConvBNRelu(16, 32),
                                          PoolingBNRelu(32))
        self.conv_IR_ch14 = nn.Sequential(ConvBNRelu(2, 16),
                                          PoolingBNRelu(16),
                                          ConvBNRelu(16, 32),
                                          PoolingBNRelu(32))
        self.conv_IR_ch15 = nn.Sequential(ConvBNRelu(2, 16),
                                          PoolingBNRelu(16),
                                          ConvBNRelu(16, 32),
                                          PoolingBNRelu(32))
        self.conv_CW = nn.Sequential(ConvBNRelu(2, 16),
                                     PoolingBNRelu(16),
                                     ConvBNRelu(16, 32),
                                     PoolingBNRelu(32))
        self.conv_CI = nn.Sequential(ConvBNRelu(2, 16),
                                     PoolingBNRelu(16),
                                     ConvBNRelu(16, 32),
                                     PoolingBNRelu(32))

        # Decoder network (224 ch)
        self.decoder_RR = nn.Sequential(DeconvConvConv(224, 64),
                                        DeconvConvConv(64, 128),
                                        nn.Conv2d(128, 128, 3, 1, 1),
                                        nn.ReLU()
                                        )

        self.decoder_RM = nn.Sequential(DeconvConvConv(224, 64),
                                        DeconvConvConv(64, 128),
                                        nn.Conv2d(128, 128, 3, 1, 1),
                                        nn.ReLU()
                                        )

        self.decoder_CW = nn.Sequential(DeconvConvConv(224, 64),
                                        DeconvConvConv(64, 128),
                                        nn.Conv2d(128, 128, 3, 1, 1),
                                        nn.ReLU()
                                        )

        self.decoder_CI = nn.Sequential(DeconvConvConv(224, 64),
                                        DeconvConvConv(64, 128),
                                        nn.Conv2d(128, 128, 3, 1, 1),
                                        nn.ReLU()
                                        )
        # Output layer
        self.out_RR = nn.Sequential(ConvReluConv(128, 1),
                                    nn.ReLU()
                                    )

        self.out_RM = nn.Sequential(ConvReluConv(128, 1),
                                    nn.Sigmoid()
                                    )
        self.out_CW = nn.Sequential(ConvReluConv(128, 1),
                                    nn.ReLU()
                                    )

        self.out_CI = nn.Sequential(ConvReluConv(128, 1),
                                    nn.ReLU()
                                    )

    def forward(self, x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw, x_ci):
        x_ir_ch08 = self.conv_IR_ch08(x_ir_ch08)
        x_ir_ch10 = self.conv_IR_ch10(x_ir_ch10)
        x_ir_ch11 = self.conv_IR_ch11(x_ir_ch11)
        x_ir_ch14 = self.conv_IR_ch14(x_ir_ch14)
        x_ir_ch15 = self.conv_IR_ch15(x_ir_ch15)
        x_cw = self.conv_CW(x_cw)
        x_ci = self.conv_CI(x_ci)
        x = torch.cat((x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw, x_ci), dim=1)
        # Decoder
        x_rainrate = self.decoder_RR(x)
        x_rainmask = self.decoder_RM(x)
        x_cw = self.decoder_CW(x)
        x_ci = self.decoder_CI(x)
        # Output
        x_rainrate = self.out_RR(x_rainrate)
        x_rainmask = self.out_RM(x_rainmask)
        x_cw = self.out_CW(x_cw)
        x_ci = self.out_CI(x_ci)
        return x_rainrate, x_rainmask, x_cw, x_ci
