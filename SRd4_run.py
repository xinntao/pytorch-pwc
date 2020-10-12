import getopt
import math
import os
import os.path as osp
import sys

import mmcv
import mmcv.video as mmvideo
import numpy as np
import PIL
import PIL.Image
import torch

try:
    from correlation import correlation  # the custom cost volume layer
except:
    sys.path.insert(0, './correlation')
    import correlation  # you should consider upgrading python
# end

##########################################################

# assert (int(str('').join(torch.__version__.split('.')[0:3])) >= 40
#         )  # requires at least pytorch version 0.4.0

torch.set_grad_enabled(
    False)  # make sure to not compute gradients for computational performance

# torch.cuda.device(1)  # change this if you have a multiple graphics cards and you want to utilize them

torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance

##########################################################

arguments_strModel = 'default'
arguments_strFirst = './images/first.png'
arguments_strSecond = './images/second.png'
arguments_strOut = './out.flo'

for strOption, strArgument in getopt.getopt(
        sys.argv[1:], '',
    [strParameter[2:] + '=' for strParameter in sys.argv[1::2]])[0]:
    if strOption == '--model' and strArgument != '':
        arguments_strModel = strArgument  # which model to use
    if strOption == '--first' and strArgument != '':
        arguments_strFirst = strArgument  # path to the first frame
    if strOption == '--second' and strArgument != '':
        arguments_strSecond = strArgument  # path to the second frame
    if strOption == '--out' and strArgument != '':
        arguments_strOut = strArgument  # path to where the output should be stored
# end

##########################################################

Backward_tensorGrid = {}
Backward_tensorPartial = {}


def Backward(tensorInput, tensorFlow):
    if str(tensorFlow.size()) not in Backward_tensorGrid:
        tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(
            1, 1, 1, tensorFlow.size(3)).expand(
                tensorFlow.size(0), -1, tensorFlow.size(2), -1)
        tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(
            1, 1, tensorFlow.size(2),
            1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))

        Backward_tensorGrid[str(tensorFlow.size())] = torch.cat(
            [tensorHorizontal, tensorVertical], 1).cuda()
    # end

    if str(tensorFlow.size()) not in Backward_tensorPartial:
        Backward_tensorPartial[str(tensorFlow.size())] = tensorFlow.new_ones(
            [tensorFlow.size(0), 1,
             tensorFlow.size(2),
             tensorFlow.size(3)])
    # end

    tensorFlow = torch.cat([
        tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0),
        tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)
    ], 1)
    tensorInput = torch.cat(
        [tensorInput, Backward_tensorPartial[str(tensorFlow.size())]], 1)

    tensorOutput = torch.nn.functional.grid_sample(
        input=tensorInput,
        grid=(Backward_tensorGrid[str(tensorFlow.size())] +
              tensorFlow).permute(0, 2, 3, 1),
        mode='bilinear',
        padding_mode='zeros')

    tensorMask = tensorOutput[:, -1:, :, :]
    tensorMask[tensorMask > 0.999] = 1.0
    tensorMask[tensorMask < 1.0] = 0.0

    return tensorOutput[:, :-1, :, :] * tensorMask


# end

##########################################################


class Network(torch.nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        class Extractor(torch.nn.Module):

            def __init__(self):
                super(Extractor, self).__init__()

                self.moduleOne = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=3,
                        out_channels=16,
                        kernel_size=3,
                        stride=2,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=16,
                        out_channels=16,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=16,
                        out_channels=16,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

                self.moduleTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=16,
                        out_channels=32,
                        kernel_size=3,
                        stride=2,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=32,
                        out_channels=32,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=32,
                        out_channels=32,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

                self.moduleThr = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=32,
                        out_channels=64,
                        kernel_size=3,
                        stride=2,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=64,
                        out_channels=64,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=64,
                        out_channels=64,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

                self.moduleFou = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=64,
                        out_channels=96,
                        kernel_size=3,
                        stride=2,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=96,
                        out_channels=96,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=96,
                        out_channels=96,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

                self.moduleFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=96,
                        out_channels=128,
                        kernel_size=3,
                        stride=2,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=128,
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=128,
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

                self.moduleSix = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=128,
                        out_channels=196,
                        kernel_size=3,
                        stride=2,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=196,
                        out_channels=196,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=196,
                        out_channels=196,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

            # end

            def forward(self, tensorInput):
                tensorOne = self.moduleOne(tensorInput)
                tensorTwo = self.moduleTwo(tensorOne)
                tensorThr = self.moduleThr(tensorTwo)
                tensorFou = self.moduleFou(tensorThr)
                tensorFiv = self.moduleFiv(tensorFou)
                tensorSix = self.moduleSix(tensorFiv)

                return [
                    tensorOne, tensorTwo, tensorThr, tensorFou, tensorFiv,
                    tensorSix
                ]

            # end

        # end

        class Decoder(torch.nn.Module):

            def __init__(self, intLevel):
                super(Decoder, self).__init__()

                intPrevious = [
                    None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2,
                    81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None
                ][intLevel + 1]
                intCurrent = [
                    None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2,
                    81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None
                ][intLevel + 0]

                if intLevel < 6:
                    self.moduleUpflow = torch.nn.ConvTranspose2d(
                        in_channels=2,
                        out_channels=2,
                        kernel_size=4,
                        stride=2,
                        padding=1)
                if intLevel < 6:
                    self.moduleUpfeat = torch.nn.ConvTranspose2d(
                        in_channels=intPrevious + 128 + 128 + 96 + 64 + 32,
                        out_channels=2,
                        kernel_size=4,
                        stride=2,
                        padding=1)
                if intLevel < 6:
                    self.dblBackward = [
                        None, None, None, 5.0, 2.5, 1.25, 0.625, None
                    ][intLevel + 1]

                self.moduleOne = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=intCurrent,
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

                self.moduleTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=intCurrent + 128,
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

                self.moduleThr = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=intCurrent + 128 + 128,
                        out_channels=96,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

                self.moduleFou = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=intCurrent + 128 + 128 + 96,
                        out_channels=64,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

                self.moduleFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=intCurrent + 128 + 128 + 96 + 64,
                        out_channels=32,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

                self.moduleSix = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=intCurrent + 128 + 128 + 96 + 64 + 32,
                        out_channels=2,
                        kernel_size=3,
                        stride=1,
                        padding=1))

            # end

            def forward(self, tensorFirst, tensorSecond, objectPrevious):
                tensorFlow = None
                tensorFeat = None

                if objectPrevious is None:
                    tensorFlow = None
                    tensorFeat = None

                    tensorVolume = torch.nn.functional.leaky_relu(
                        input=correlation.FunctionCorrelation(
                            tensorFirst=tensorFirst,
                            tensorSecond=tensorSecond),
                        negative_slope=0.1,
                        inplace=False)

                    tensorFeat = torch.cat([tensorVolume], 1)

                elif objectPrevious is not None:
                    tensorFlow = self.moduleUpflow(
                        objectPrevious['tensorFlow'])
                    tensorFeat = self.moduleUpfeat(
                        objectPrevious['tensorFeat'])

                    tensorVolume = torch.nn.functional.leaky_relu(
                        input=correlation.FunctionCorrelation(
                            tensorFirst=tensorFirst,
                            tensorSecond=Backward(
                                tensorInput=tensorSecond,
                                tensorFlow=tensorFlow * self.dblBackward)),
                        negative_slope=0.1,
                        inplace=False)

                    tensorFeat = torch.cat(
                        [tensorVolume, tensorFirst, tensorFlow, tensorFeat], 1)

                # end

                tensorFeat = torch.cat(
                    [self.moduleOne(tensorFeat), tensorFeat], 1)
                tensorFeat = torch.cat(
                    [self.moduleTwo(tensorFeat), tensorFeat], 1)
                tensorFeat = torch.cat(
                    [self.moduleThr(tensorFeat), tensorFeat], 1)
                tensorFeat = torch.cat(
                    [self.moduleFou(tensorFeat), tensorFeat], 1)
                tensorFeat = torch.cat(
                    [self.moduleFiv(tensorFeat), tensorFeat], 1)

                tensorFlow = self.moduleSix(tensorFeat)

                return {'tensorFlow': tensorFlow, 'tensorFeat': tensorFeat}

            # end

        # end

        class Refiner(torch.nn.Module):

            def __init__(self):
                super(Refiner, self).__init__()

                self.moduleMain = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32,
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        dilation=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=128,
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=2,
                        dilation=2),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=128,
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=4,
                        dilation=4),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=128,
                        out_channels=96,
                        kernel_size=3,
                        stride=1,
                        padding=8,
                        dilation=8),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=96,
                        out_channels=64,
                        kernel_size=3,
                        stride=1,
                        padding=16,
                        dilation=16),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=64,
                        out_channels=32,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        dilation=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=32,
                        out_channels=2,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        dilation=1))

            # end

            def forward(self, tensorInput):
                return self.moduleMain(tensorInput)

            # end

        # end

        self.moduleExtractor = Extractor()

        self.moduleTwo = Decoder(2)
        self.moduleThr = Decoder(3)
        self.moduleFou = Decoder(4)
        self.moduleFiv = Decoder(5)
        self.moduleSix = Decoder(6)

        self.moduleRefiner = Refiner()

        self.load_state_dict(
            torch.load('./network-' + arguments_strModel + '.pytorch'))

    # end

    def forward(self, tensorFirst, tensorSecond):
        tensorFirst = self.moduleExtractor(tensorFirst)  # 6 pyramid levels
        tensorSecond = self.moduleExtractor(tensorSecond)

        objectEstimate = self.moduleSix(tensorFirst[-1], tensorSecond[-1],
                                        None)
        objectEstimate = self.moduleFiv(tensorFirst[-2], tensorSecond[-2],
                                        objectEstimate)
        objectEstimate = self.moduleFou(tensorFirst[-3], tensorSecond[-3],
                                        objectEstimate)
        objectEstimate = self.moduleThr(tensorFirst[-4], tensorSecond[-4],
                                        objectEstimate)
        objectEstimate = self.moduleTwo(tensorFirst[-5], tensorSecond[-5],
                                        objectEstimate)

        return objectEstimate['tensorFlow'] + self.moduleRefiner(
            objectEstimate['tensorFeat'])

    # end


# end

moduleNetwork = Network().cuda().eval()

##########################################################


def estimate(tensorFirst, tensorSecond):
    tensorOutput = torch.FloatTensor()

    assert (tensorFirst.size(1) == tensorSecond.size(1))
    assert (tensorFirst.size(2) == tensorSecond.size(2))

    intWidth = tensorFirst.size(2)
    intHeight = tensorFirst.size(1)

    # assert (
    #     intWidth == 1024
    # )  # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    # assert (
    #     intHeight == 436
    # )  # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

    if True:
        tensorFirst = tensorFirst.cuda()
        tensorSecond = tensorSecond.cuda()
        tensorOutput = tensorOutput.cuda()
    # end

    if True:
        tensorPreprocessedFirst = tensorFirst.view(1, 3, intHeight, intWidth)
        tensorPreprocessedSecond = tensorSecond.view(1, 3, intHeight, intWidth)

        intPreprocessedWidth = int(
            math.floor(math.ceil(intWidth / 64.0) * 64.0))
        intPreprocessedHeight = int(
            math.floor(math.ceil(intHeight / 64.0) * 64.0))

        tensorPreprocessedFirst = torch.nn.functional.interpolate(
            input=tensorPreprocessedFirst,
            size=(intPreprocessedHeight, intPreprocessedWidth),
            mode='bilinear',
            align_corners=False)
        tensorPreprocessedSecond = torch.nn.functional.interpolate(
            input=tensorPreprocessedSecond,
            size=(intPreprocessedHeight, intPreprocessedWidth),
            mode='bilinear',
            align_corners=False)

        tensorFlow = 20.0 * torch.nn.functional.interpolate(
            input=moduleNetwork(tensorPreprocessedFirst,
                                tensorPreprocessedSecond),
            size=(intHeight, intWidth),
            mode='bilinear',
            align_corners=False)

        tensorFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
        tensorFlow[:,
                   1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

        tensorOutput.resize_(2, intHeight,
                             intWidth).copy_(tensorFlow[0, :, :, :])
    # end

    # end

    return tensorOutput


def run_once(img1, img2):
    # calculate optical flow from img1 to img2
    tensorFirst = torch.FloatTensor(
        img1[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0))
    tensorSecond = torch.FloatTensor(
        img2[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0))

    tensorOutput = estimate(tensorFirst, tensorSecond)  # 2*H*W

    # downsample flow
    # C, H, W = tensorOutput.size()
    # tensorOutput[0, :, :] *= 1 / 4
    # tensorOutput[1, :, :] *= 1 / 4

    # tensorOutput = torch.nn.functional.interpolate(
    #     input=tensorOutput.unsqueeze_(0),
    #     size=(H // 4, W // 4),
    #     mode='bilinear',
    #     align_corners=False)

    if True:
        tensorOutput = tensorOutput.squeeze(0).cpu()

    return tensorOutput

    # tensor to numpy
    # tensorOutput = tensorOutput.numpy().transpose(1, 2, 0)
    # mmvideo.flowwrite(tensorOutput, save_name, quantize=True, concat_axis=0, max_val=20, norm=False)


##########################################################

if __name__ == '__main__':
    import glob

    import cv2
    data_root = 'high_res_motion_transfer/256_testdata/celeba_00000000'
    ref_path = 'high_res_motion_transfer/celeba_00000000_256.png'
    ref_path_512 = 'high_res_motion_transfer/celeba_00000000_512.png'

    save_flow_folder = 'results/flow'
    save_warped_folder = 'results/warp'
    save_warped_folder_512 = 'results/warp_512'
    mmcv.utils.mkdir_or_exist(save_flow_folder)
    mmcv.utils.mkdir_or_exist(save_warped_folder)
    mmcv.utils.mkdir_or_exist(save_warped_folder_512)

    img_ref = cv2.imread(ref_path)
    img_ref_512 = cv2.imread(ref_path_512)
    # whether bicubic upsample
    # h, w, _ = img_ref.shape
    # img_ref = cv2.resize(
    #     img_ref, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)

    img_paths = sorted(glob.glob(os.path.join(data_root, '*')))
    for idx, img_path in enumerate(img_paths):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        print(idx, basename)
        # read image
        img_input = cv2.imread(img_path)
        # whether bicubic upsample
        # h, w, _ = img_input.shape
        # img_input = cv2.resize(
        #     img_input, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
        # caulcate flow from img_input (e.g., 049) to img_ref (050)
        flow_tensor = run_once(img_input, img_ref)  # [2, h, w]

        # downsample flow
        # c, h, w = flow_tensor.size()
        # flow_tensor = torch.nn.functional.interpolate(
        #     input=flow_tensor.unsqueeze_(0),
        #     size=(h // 4, w // 4),
        #     mode='bicubic',
        #     align_corners=False)
        # flow_tensor = flow_tensor / 4
        # flow_tensor.squeeze_(0)

        # save flow
        # tensor to numpy
        flow_np = flow_tensor.numpy().transpose(1, 2, 0)  # [h, w, 2]
        flow_np = np.ascontiguousarray(flow_np, dtype=np.float32)
        flow_vis = mmcv.visualization.optflow.flow2rgb(flow_np)
        mmcv.imwrite(flow_vis * 255,
                     osp.join(save_flow_folder, f'{basename}_flow.png'))
        # warp
        warped_img = mmcv.video.flow_warp(
            img_ref, flow_np, filling_value=0, interpolate_mode='bilinear')
        # save warped images
        mmcv.imwrite(warped_img,
                     osp.join(save_warped_folder, f'{basename}_warped.png'))

        c, h, w = flow_tensor.size()
        flow_tensor = torch.nn.functional.interpolate(
            input=flow_tensor.unsqueeze_(0),
            size=(h * 2, w * 2),
            mode='bicubic',
            align_corners=False)
        flow_tensor = flow_tensor * 2
        flow_tensor.squeeze_(0)
        flow_np = flow_tensor.numpy().transpose(1, 2, 0)  # [h, w, 2]
        flow_np = np.ascontiguousarray(flow_np, dtype=np.float32)
        # warp
        warped_img_512 = mmcv.video.flow_warp(
            img_ref_512, flow_np, filling_value=0, interpolate_mode='bilinear')
        # save warped images
        mmcv.imwrite(
            warped_img,
            osp.join(save_warped_folder_512, f'{basename}_warped.png'))
