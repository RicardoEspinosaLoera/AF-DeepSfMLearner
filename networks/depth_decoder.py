# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales= range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

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
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

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
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#from collections import OrderedDict
from layers import *
from typing import Dict, Tuple, List

class DepthDecoder(nn.Module):
    #convs:Dict[str,int,int]
    def __init__(self, num_ch_enc, scales= range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = [16, 32, 64, 128, 256]
        #self.output : Dict[Tuple[string,int,int],torch.Tensor] = {} 
        self.output : Dict[string,torch.Tensor] = {} 
        #self.output = []
        # decoder

        #i = 4
        num_ch_in = self.num_ch_enc[-1]
        num_ch_out = self.num_ch_dec[4]
        self.convs4_0 = ConvBlock(int(num_ch_in), int(num_ch_out))

        num_ch_in = self.num_ch_dec[4]
        num_ch_in += self.num_ch_enc[3]
        num_ch_out = self.num_ch_dec[4]
        self.convs4_1 = ConvBlock(int(num_ch_in), int(num_ch_out))

         #i = 3
        num_ch_in = self.num_ch_dec[4]
        num_ch_out = self.num_ch_dec[3]
        self.convs3_0 = ConvBlock(int(num_ch_in), int(num_ch_out))

        num_ch_in = self.num_ch_dec[3]
        num_ch_in += self.num_ch_enc[2]
        num_ch_out = self.num_ch_dec[3]
        self.convs3_1 = ConvBlock(int(num_ch_in), int(num_ch_out))

         #i = 2

        num_ch_in = self.num_ch_dec[3]
        num_ch_out = self.num_ch_dec[2]
        self.convs2_0 = ConvBlock(int(num_ch_in), int(num_ch_out))

        num_ch_in = self.num_ch_dec[2]
        num_ch_in += self.num_ch_enc[1]
        num_ch_out = self.num_ch_dec[2]
        self.convs2_1 = ConvBlock(int(num_ch_in), int(num_ch_out))

        #i = 1
        num_ch_in = self.num_ch_dec[2]
        num_ch_out = self.num_ch_dec[1]
        self.convs1_0 = ConvBlock(int(num_ch_in), int(num_ch_out))

        num_ch_in = self.num_ch_dec[1]
        num_ch_in += self.num_ch_enc[1]
        num_ch_out = self.num_ch_dec[1]
        self.convs1_1 = ConvBlock(int(num_ch_in), int(num_ch_out))

        #i = 0

        num_ch_in = self.num_ch_dec[1]
        num_ch_out = self.num_ch_dec[0]
        self.convs0_0 = ConvBlock(int(num_ch_in), int(num_ch_out))

        num_ch_in = self.num_ch_dec[0]
        num_ch_out = self.num_ch_dec[0]
        self.convs0_1 = ConvBlock(int(num_ch_in), int(num_ch_out))

        self.dispconv0 = Conv3x3(int(self.num_ch_dec[0]), int(self.num_output_channels))
        self.dispconv1 = Conv3x3(int(self.num_ch_dec[1]), int(self.num_output_channels))
        self.dispconv2 = Conv3x3(int(self.num_ch_dec[2]), int(self.num_output_channels))
        self.dispconv3 = Conv3x3(int(self.num_ch_dec[3]), int(self.num_output_channels))

        
        #self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self,input_features: List[torch.Tensor]):
        #self.output = []
        # decoder
        x = input_features[-1]
        #print(x.shape)
        x = self.convs4_0(x)
        x = [F.interpolate(x, scale_factor=float(2), mode="nearest")]
        #x = [upsample(x)]
        x += [input_features[3]]
        x = torch.cat(x, 1)
        x = self.convs4_1(x)      
        
        x = self.convs3_0(x)
        x = [F.interpolate(x, scale_factor=float(2), mode="nearest")]
        #x = [upsample(x)]
        x += [input_features[2]]
        x = torch.cat(x, 1)
        x = self.convs3_1(x)
        #servers: Sequence[tuple[tuple[str, int], dict[str, str]]])
        self.output["disp_3"] = self.sigmoid(self.dispconv3(x))

        x = self.convs2_0(x)
        #x = [upsample(x)]
        x = [F.interpolate(x, scale_factor=float(2), mode="nearest")]
        x += [input_features[1]]
        x = torch.cat(x, 1)
        x = self.convs2_1(x)
        self.output["disp_2"] = self.sigmoid(self.dispconv2(x))

        x = self.convs1_0(x)
        #x = [upsample(x)]
        x = [F.interpolate(x, scale_factor=float(2), mode="nearest")]
        x += [input_features[0]]
        x = torch.cat(x, 1)
        x = self.convs1_1(x)
        self.output["disp_1"] = self.sigmoid(self.dispconv1(x))

        x = self.convs0_0(x)
        #x = [upsample(x)]
        x = [F.interpolate(x, scale_factor=float(2), mode="nearest")]
        x = torch.cat(x, 1)
        x = self.convs0_1(x)
        self.output["disp_0"] = self.sigmoid(self.dispconv0(x))        

        return self.output
