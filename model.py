import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init
from region_nl import RegionNONLocalBlock

torch.cuda.set_device(0)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class conv_relu(nn.Module):
    def __init__(self, inchannels, outchannels, stride=1):
        super(conv_relu, self).__init__()
        self.channel = outchannels // 2
        self.layer1 = nn.Sequential(
            nn.Conv2d(inchannels, self.channel, 3, stride, 1),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(inchannels, self.channel, 3, stride, 1),
            nn.ReLU(inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(self.channel, self.channel // 2, (3, 1), stride, (1, 0)),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(self.channel, self.channel // 2, (1, 3), stride, (0, 1)),
            nn.ReLU(inplace=True)
        )



        self.ca = ChannelAttention(outchannels)

    def forward(self, x):
        layer1 = self.layer1(x)
        layer2 = self.layer2(x)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer2)
        concat = torch.cat((layer1, layer3, layer4), dim=1)
        concat = self.ca(concat) * concat

        return concat


class Denoising_Block(nn.Module):
    def __init__(self ,block_num=5 ,inter_channel=64, channel=96):
        super(Denoising_Block, self).__init__()

        channels_now = channel +inter_channel
        self.group_list =[]
        self.conv1 =conv_relu(channel ,inter_channel)

        for i in range(block_num -1):

            group = nn.Sequential(
                nn.Conv2d(channels_now, inter_channel, 1, 1, 0),
                nn.ReLU(),
                conv_relu(inter_channel ,inter_channel)
            )
            self.add_module(name='group_%d' % i, module=group)
            self.group_list.append(group)
            channels_now += inter_channel



        self.ca1 = ChannelAttention(channel)

        self.conv_concat = nn.Conv2d(inter_channel *block_num +channel, channel, 1, 1, 0)

    def forward(self, input):
        conv1 =self.conv1(input)
        feature_list = [input ,conv1]
        for group in self.group_list:
            inputs = torch.cat(feature_list, dim=1)
            outputs = group(inputs)
            feature_list.append(outputs)
        concat = torch.cat(feature_list, dim=1)
        out =self.conv_concat(concat)
        out =self.ca1(out) *out +input


        return out


class NCB(nn.Module):
    def __init__(self, inchannels, outchannels, grid=[4, 4]):
        super(NCB, self).__init__()
        self.ca1 = ChannelAttention(inchannels)
        self.encab = RegionNONLocalBlock(inchannels, grid=grid)
        self.conv = conv_relu(inchannels, outchannels)



    def forward(self, x):
        encab = self.encab(x)
        ca1 = self.ca1(encab) * encab
        conv = self.conv(ca1)

        return conv + x


class ENCAM(nn.Module):
    def __init__(self ,channels=96):
        super(ENCAM, self).__init__()

        self.f2d_3 = nn.Conv2d(1, 32, 3, 1, 1)
        self.f2d_5_1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.f2d_5_2 = nn.Conv2d(32, 16, (3, 1), 1, (1, 0))
        self.f2d_5_3 = nn.Conv2d(32, 16, (1, 3), 1, (0, 1))

        self.f3d_1 = nn.Conv3d(1, 16, (30, 3, 3), 1, (0, 1, 1))
        self.f3d_2 = nn.Conv3d(1, 16, (30, 7, 7), 1, (0, 3, 3))

        self.NCB_1 = NCB(channels, channels, grid=[6, 6])
        self.NCB_2 = NCB(channels, channels, grid=[4, 4])
        self.NCB_3 = NCB(channels, channels, grid=[2, 2])
        self.NCB_4 = NCB(channels, channels, grid=[2, 2])
        self.NCB_5 = NCB(channels, channels, grid=[4, 4])
        self.NCB_6 = NCB(channels, channels, grid=[6, 6])

        self.Dblock1 = Denoising_Block()
        self.Dblock2 = Denoising_Block()
        self.Dblock3 = Denoising_Block()
        self.Dblock4 = Denoising_Block()
        self.Dblock5 = Denoising_Block()
        self.Dblock6 = Denoising_Block()

        self.concat = nn.Sequential(
            nn.Conv2d(672, 1, 3, 1, 1),

        )
        self._initialize_weights()

        self.ca = ChannelAttention(672)

    def forward(self, x, y):
        f2d_3 = self.f2d_3(x)
        f2d_5_1 = self.f2d_5_1(x)
        f2d_5_2 = self.f2d_5_2(F.relu(f2d_5_1))
        f2d_5_3 = self.f2d_5_3(F.relu(f2d_5_1))

        f3d_1 = self.f3d_1(y)
        f3d_2 = self.f3d_2(y)

        out1 = F.relu(torch.cat((f2d_3, f2d_5_2, f2d_5_3), dim=1))
        out2 = F.relu(torch.cat((f3d_1, f3d_2), dim=1)).squeeze(2)
        out3 = torch.cat((out1, out2), dim=1)

        block1 = self.Dblock1(self.NCB_1(out3))
        block2 = self.Dblock2(self.NCB_2(out3 +block1))
        block3 = self.Dblock3(self.NCB_3(out3 +block2))
        block4 = self.Dblock4(self.NCB_4(out3 +block3))
        block5 = self.Dblock5(self.NCB_5(out3 +block4))
        block6 = self.Dblock6(self.NCB_6(out3 +block5))


        concat = torch.cat(
            (out3, block1, block2, block3, block4, block5, block6), dim=1)
        concat = concat * self.ca(concat)
        concat = self.concat(concat)


        return x - concat

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)

                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                init.orthogonal_(m.weight)
                # print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


def model_5():
    models=ENCAM()
    models.load_state_dict(torch.load('ENCAM_SIGMA005.pth',map_location='cuda:0'))
    models.cuda()
    models.eval()
    return models

def model_25():
    models=ENCAM()
    models.load_state_dict(torch.load('ENCAM_SIGMA025.pth',map_location='cuda:0'))
    models.cuda()
    models.eval()
    return models

def model_50():
    models=ENCAM()
    models.load_state_dict(torch.load('ENCAM_SIGMA050.pth',map_location='cuda:0'))
    models.cuda()
    models.eval()
    return models
def model_75():
    models=ENCAM()
    models.load_state_dict(torch.load('ENCAM_SIGMA075.pth',map_location='cuda:0'))
    models.cuda()
    models.eval()
    return models
def model_100():
    models=ENCAM()
    models.load_state_dict(torch.load('ENCAM_SIGMA100.pth',map_location='cuda:0'))
    models.cuda()
    models.eval()
    return models
def model_real():
    # 'model_mnet6RES_048SIGMA055.pth', map_location = 'cuda:0
    models=ENCAM()
    models.load_state_dict(torch.load('ENCAM_rand_SIGMA075.pth',map_location='cuda:0'))
    models.cuda()
    models.eval()
    return models

def tests(tests,model):
    tests=np.array(tests)

    tests=tests.reshape(200,200,191)

    k=15
    cs = tests.shape[2]
    print(cs)
    tests = tests.astype(np.float32)
    tests=tests.transpose((2,1,0))
    test_out = np.zeros(tests.shape).astype(np.float32)

    for channel_i in range(cs):
        x = tests[channel_i, :, :]

        x = np.expand_dims(x, axis=0)
        x = np.expand_dims(x, axis=0)
        if channel_i < k:
            # print(channel_i)
            y = tests[0:30, :, :]


        elif channel_i < cs - k:

            y = np.concatenate((tests[channel_i - k:channel_i, :, :],
                                tests[channel_i + 1:channel_i + k + 1, :, :]))

            # print(y.shape)
        else:
            y = tests[cs - 30:cs, :, :]

        y = np.expand_dims(y, axis=0)
        y = np.expand_dims(y, axis=0)
        # x=x.astype('float32')/255.0
        # y=y.astype('float32')/255.0
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        # x=x+torch.randn(x.size()).mul_(SIGMA/255.0)
        # y=y+torch.randn(y.size()).mul_(SIGMA/255.0)
        with torch.no_grad():

            out = model(x.cuda(), y.cuda())

        out = out.squeeze(0)

        out = out.data.cpu().numpy()

        test_out[channel_i, :, :] = out
    test_out = test_out.transpose((2, 1, 0))
    test_out=test_out.astype(np.double)
    return test_out

