import torch
import os
import numpy as np
import argparse
import cv2
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as vutils
from network.Transformer import Transformer
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default = 'test_img')
parser.add_argument('--load_size', default = 450)
parser.add_argument('--model_path', default = './pretrained_model')
parser.add_argument('--style', default = 'Hayao')
parser.add_argument('--output_dir', default = 'test_output')
parser.add_argument('--gpu', type=int, default = 0)
parser.add_argument('--video', default='none')

opt = parser.parse_args()

valid_ext = ['.jpg', '.png', 'jpeg']

if not os.path.exists(opt.output_dir): os.mkdir(opt.output_dir)

# load pretrained model
model = Transformer()
model.load_state_dict(torch.load(os.path.join(opt.model_path, opt.style + '_net_G_float.pth')))
model.eval()

if opt.gpu > -1:
    print('GPU mode')
    model.cuda()
else:
    print('CPU mode')
    model.float()

count = 0

if(opt.video != 'none'):
    vid = cv2.VideoCapture(opt.video)
    if(vid.isOpened()==False):
        print("Error opening stream or file")
        print("Recheck the file address: {}".format(opt.video))
    else:
        frame_width = int(vid.get(3))
        frame_height = int(vid.get(4))
        out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
        while(vid.isOpened()):
            ret, frame = vid.read()
            count += 1
            if(ret == True and count % 20 == 0):
                # frame
                # input_image = cv2.imread(os.path.join(opt.input_dir, files), 1)
                input_image = frame
                # resize image, keep aspect ratio
                h = input_image.shape[0]
                w = input_image.shape[1]
                # h = input_image.size[0]
                # w = input_image.size[1]
                ratio = h *1.0 / w
                if ratio > 1:
                    h = opt.load_size
                    w = int(h*1.0/ratio)
                else:
                    w = opt.load_size
                    h = int(w * ratio)
                input_image = cv2.resize(input_image, (w, h), cv2.INTER_CUBIC)
                # input_image = input_image.resize((h, w), Image.BICUBIC)
                input_image = np.asarray(input_image)
                # RGB -> BGR
                input_image = input_image[:, :, [2, 1, 0]]
                input_image = transforms.ToTensor()(input_image).unsqueeze(0)
                # preprocess, (-1, 1)
                input_image = -1 + 2 * input_image 
                if opt.gpu > -1:
                    input_image = Variable(input_image, volatile=True).cuda()
                else:
                    input_image = Variable(input_image, volatile=True).float()
                # forward
                output_image = model(input_image)
                output_image = output_image[0]
                # BGR -> RGB
                # output_image = output_image[[2, 1, 0], :, :]
                # deprocess, (0, 1)
                output_image = output_image.data.cpu().float() * 0.5 + 0.5
                # save
                out.write(np.uint8(output_image))
                # vutils.save_image(output_image, os.path.join(opt.video, files[:-4] + '_' + opt.style + '.jpg'))
            out.release()
            print('Done!')