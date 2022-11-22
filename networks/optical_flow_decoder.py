"""from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from torch.distributions.normal import Normal
from collections import OrderedDict
from layers import *


class PositionDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales = range(4) , num_output_channels=2, use_skips=True):
        super(PositionDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.conv = getattr(nn, 'Conv2d')

        # decoder
        self.convs = OrderedDict() # 有序字典
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:

            self.convs[("position_conv", s)] = self.conv (self.num_ch_dec[s], self.num_output_channels, kernel_size = 3, padding = 1)
            # init flow layer with small weights and bias
            self.convs[("position_conv", s)].weight = nn.Parameter(Normal(0, 1e-5).sample(self.convs[("position_conv", s)].weight.shape))
            self.convs[("position_conv", s)].bias = nn.Parameter(torch.zeros(self.convs[("position_conv", s)].bias.shape))

        self.decoder = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        self.outputs = {}
        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("position", i)] = self.convs[("position_conv", i)](x)

        return self.outputs
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from torch.distributions.normal import Normal
#from collections import OrderedDict
from typing import Dict, Tuple
from layers import *


class PositionDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales = range(4) , num_output_channels=2, use_skips=True):
        super(PositionDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = [16, 32, 64, 128, 256]
        self.conv = getattr(nn, 'Conv2d')


        self.outputs : Dict[string,torch.Tensor] = {} 
        #self.output : Dict[Tuple[string,int,int],torch.Tensor] = {} 
        #self.output : Dict[string,torch.Tensor] = {} 
        #self.output = []
        # decoder

        #i = 4
        num_ch_in = self.num_ch_enc[-1]
        num_ch_out = self.num_ch_dec[4]
        self.convs4_0 = ConvBlock(num_ch_in, num_ch_out)

        num_ch_in = self.num_ch_dec[4]
        num_ch_in += self.num_ch_enc[3]
        num_ch_out = self.num_ch_dec[4]
        self.convs4_1 = ConvBlock(num_ch_in, num_ch_out)

        #i = 3
        num_ch_in = self.num_ch_dec[4]
        num_ch_out = self.num_ch_dec[3]
        self.convs3_0 = ConvBlock(num_ch_in, num_ch_out)

        num_ch_in = self.num_ch_dec[3]
        num_ch_in += self.num_ch_enc[2]
        num_ch_out = self.num_ch_dec[3]
        self.convs3_1 = ConvBlock(num_ch_in, num_ch_out)

        #i = 2
        num_ch_in = self.num_ch_dec[3]
        num_ch_out = self.num_ch_dec[2]
        self.convs2_0 = ConvBlock(num_ch_in, num_ch_out)

        num_ch_in = self.num_ch_dec[2]
        num_ch_in += self.num_ch_enc[1]
        num_ch_out = self.num_ch_dec[2]
        self.convs2_1 = ConvBlock(num_ch_in,num_ch_out)

        #i = 1
        num_ch_in = self.num_ch_dec[2]
        num_ch_out = self.num_ch_dec[1]
        self.convs1_0 = ConvBlock(num_ch_in, num_ch_out)

        num_ch_in = self.num_ch_dec[1]
        num_ch_in += self.num_ch_enc[1]
        num_ch_out = self.num_ch_dec[1]
        self.convs1_1 = ConvBlock(num_ch_in, num_ch_out)

        #i = 0

        num_ch_in = self.num_ch_dec[1]
        num_ch_out = self.num_ch_dec[0]
        self.convs0_0 = ConvBlock(int(num_ch_in), int(num_ch_out))

        num_ch_in = self.num_ch_dec[0]
        num_ch_out = self.num_ch_dec[0]
        self.convs0_1 = ConvBlock(int(num_ch_in), int(num_ch_out))

        self.position_conv_0 = self.conv(self.num_ch_dec[0], self.num_output_channels, kernel_size = 3, padding = 1)
        self.position_conv_0.weight = nn.Parameter(Normal(0, 1e-5).sample(self.position_conv_0.weight.shape))
        self.position_conv_0.bias = nn.Parameter(torch.zeros(self.position_conv_0.bias.shape))

        self.position_conv_1 = self.conv(self.num_ch_dec[1], self.num_output_channels, kernel_size = 3, padding = 1)
        self.position_conv_1.weight = nn.Parameter(Normal(0, 1e-5).sample(self.position_conv_1.weight.shape))
        self.position_conv_1.bias = nn.Parameter(torch.zeros(self.position_conv_1.bias.shape))

        self.position_conv_2 = self.conv(self.num_ch_dec[2], self.num_output_channels, kernel_size = 3, padding = 1)
        self.position_conv_2.weight = nn.Parameter(Normal(0, 1e-5).sample(self.position_conv_2.weight.shape))
        self.position_conv_2.bias = nn.Parameter(torch.zeros(self.position_conv_2.bias.shape))

        self.position_conv_3 = self.conv(self.num_ch_dec[3], self.num_output_channels, kernel_size = 3, padding = 1)
        self.position_conv_3.weight = nn.Parameter(Normal(0, 1e-5).sample(self.position_conv_3.weight.shape))
        self.position_conv_3.bias = nn.Parameter(torch.zeros(self.position_conv_3.bias.shape))

       

    def forward(self, input_features):
        #self.outputs = {}
        # decoder

        x = input_features[-1]
        #print(x.shape)
        x = self.convs4_0(x)
        x = [F.interpolate(x, scale_factor=float(2), mode="nearest")]
        #x = [upsample(x)]
        x += [input_features[3]]
        x = torch.cat(x, 1)
        x = self.convs4_1(x)     
        #self.outputs["position_4"] = self.position_conv_4(x) 
        
        x = self.convs3_0(x)
        x = [F.interpolate(x, scale_factor=float(2), mode="nearest")]
        #x = [upsample(x)]
        x += [input_features[2]]
        x = torch.cat(x, 1)
        x = self.convs3_1(x)
        #self.output["disp_3"] = self.sigmoid(self.dispconv3(x))
        self.outputs["position_3"] = self.position_conv_3(x)

        x = self.convs2_0(x)
        #x = [upsample(x)]
        x = [F.interpolate(x, scale_factor=float(2), mode="nearest")]
        x += [input_features[1]]
        x = torch.cat(x, 1)
        x = self.convs2_1(x)
        #self.output["disp_2"] = self.sigmoid(self.dispconv2(x))
        self.outputs["position_2"] = self.position_conv_2(x)

        x = self.convs1_0(x)
        #x = [upsample(x)]
        x = [F.interpolate(x, scale_factor=float(2), mode="nearest")]
        x += [input_features[0]]
        x = torch.cat(x, 1)
        x = self.convs1_1(x)
        #self.output["disp_1"] = self.sigmoid(self.dispconv1(x))
        self.outputs["position_1"] = self.position_conv_1(x)

        x = self.convs0_0(x)
        #x = [upsample(x)]
        x = [F.interpolate(x, scale_factor=float(2), mode="nearest")]
        x = torch.cat(x, 1)
        x = self.convs0_1(x)
        #self.output["disp_0"] = self.sigmoid(self.dispconv0(x))
        self.outputs["position_0"] = self.position_conv_0(x)



        return self.outputs
        
