import os
import sys
import json
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.backends import cudnn
import torch.nn as nn
import torchvision
from PIL import Image
from utils.func import *
from utils.vis import *
from utils.IoU import *
from utils.augment import *
from utils.accuracy import *

import argparse
from Model import *
from skimage import measure

class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='BAS Localization evaluation')
        self.parser.add_argument('--input_size',default=256,dest='input_size')
        self.parser.add_argument('--crop_size',default=224,dest='crop_size')
        self.parser.add_argument('--phase', type=str, default='test') 
        self.parser.add_argument('--num_classes',default=100)
        self.parser.add_argument('--tencrop', default=True)
        self.parser.add_argument('--gpu',help='which gpu to use',default='0',dest='gpu')
        self.parser.add_argument('--data',metavar='DIR',default='../imagenet100',help='path to imagenet dataset')
        self.parser.add_argument('--arch', type=str, default='mobilenet')  ## choosen  [ vgg, resnet, inception, mobilenet, resnet18] 
        self.parser.add_argument('--threshold', type=float, default=0.71)
        self.parser.add_argument('--adap', type=float, default=1)
        self.parser.add_argument('--adap_loc', type=float, default=0.2)

    def parse(self):
        opt = self.parser.parse_args()
        opt.arch = opt.arch     
        return opt

args = opts().parse()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu



print(args.arch)
print(args.threshold)
print("adap:",args.adap)
print("adap_loc:",args.adap_loc)

def normalize_map(atten_map,w,h):
    min_val = np.min(atten_map)
    max_val = np.max(atten_map)
    #atten_norm = (atten_map - min_val)/(max_val - min_val)
    atten_norm = (atten_map)/(max_val)
    #atten_map_p = atten_map*0.8
    #aa=np.max(atten_map - atten_map_p)
    #atten_norm = (atten_map - atten_map_p)/aa
    atten_norm = cv2.resize(atten_norm, dsize=(w,h))
    return atten_norm

def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data  
    
cudnn.benchmark = False
TEN_CROP = args.tencrop
normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
transform = transforms.Compose([
        transforms.Resize((args.input_size,args.input_size)),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        normalize
])
cls_transform = transforms.Compose([
        transforms.Resize((args.crop_size,args.crop_size)),
        transforms.ToTensor(),
        normalize
])
ten_crop_aug = transforms.Compose([
    transforms.Resize((args.input_size, args.input_size)),
    transforms.TenCrop(args.crop_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
])
model = torch.load('./logs/mobilenet.pth.tar')
model = nn.DataParallel(model, device_ids=[0])

model = model.to(0)
model.eval()

adap = torch.load('./logs/adaptation.pth.tar')
adap = nn.DataParallel(adap, device_ids=[0])

adap = adap.to(0)
adap.eval()
#print(adap)
root = args.data
val_imagedir = os.path.join(root, 'val')


anno_root = os.path.join(root,'bbox')
val_annodir = './gt/val_gt_100.txt'
val_list_path = './gt/val_list_100.txt'

classes = os.listdir(val_imagedir)
classes.sort()
temp_softmax = nn.Softmax()
class_to_idx = {classes[i]:i for i in range(len(classes))}

accs = []
accs_top5 = []
loc_accs = []
final_loc = []
final_clsloc = []
final_clsloctop5 = []
bbox_f = open(val_annodir, 'r')
bbox_list = {}

for line in bbox_f:
    part_1, part_2 = line.strip('\n').split(';')
    temp_name, w, h, _ = part_1.split(' ')
    temp_name = temp_name[:9] + '/' + temp_name[11:]
    part_2 = part_2[1:]
    bbox = part_2.split(' ')
    bbox = np.array(bbox, dtype=np.float32)
    box_num = len(bbox) // 4
    w, h = np.float32(w),np.float32(h)
    for i in range(box_num):
        bbox[4*i], bbox[4*i+1], bbox[4*i+2], bbox[4*i+3] = bbox[4*i], bbox[4*i+1], bbox[4*i+2] , bbox[4*i+3]
    bbox_list[temp_name] = bbox  ## gt
cur_num = 0
bbox_f.close()

files = [[] for i in range(100)] 
with open(val_list_path, 'r') as f:
    for line in f:
        #print(line)
        test_img_path, img_class =  line.strip("\n").split(';')
        files[int(img_class)].append(test_img_path)
cls_acc_1 = AverageMeter()
cls_acc_5 = AverageMeter()
count=0
avg_pool = nn.AvgPool2d(14)

#print(files)
for k in range(100):
    cls = classes[k]
    IoUSet = []
    IoUSetTop5 = []
    LocSet = []

    for (i, name) in enumerate(files[k]):
        gt_boxes = bbox_list[name]
        cur_num += 1
        if len(gt_boxes)==0:
            continue

        raw_img = Image.open(os.path.join(val_imagedir, name)).convert('RGB')
        w, h = args.crop_size, args.crop_size

        with torch.no_grad():
            img = transform(raw_img)
            img = torch.unsqueeze(img, 0)
            img = img.to(0)
            adap_input, loc_adap_input = model(img, torch.tensor([class_to_idx[cls]]), inference=True)

            loc = adap(loc_adap_input) #1, 90, 


            if k< args.num_classes - 10:

                loc = loc[0][k].data.cpu()
           

            output1,_,_ = model(img, torch.tensor([class_to_idx[cls]]) ) #1, 100
            output2= output1.clone()
            
            adap_output1 = adap(adap_input, adap=True) #1, 90, 14, 14

            adap_output1 = avg_pool(adap_output1).view(adap_output1.size(0), -1) #1, 90



            front_exemplar=output1[:, :-10] + adap_output1*args.adap #64 50 14 14 + 64 50 14 14
            back_exemplar =output1[:, -10:] # 64 10 14 14

            output1 = torch.cat([front_exemplar,back_exemplar], dim=1)


            _, predicted = output1.max(1)
            _, predicted5_or = output2.topk(5, 1)
            _, predicted5 = output1.topk(5, 1)

            cur_cls_acc_1 = 100. * compute_cls_acc(output1, torch.tensor([class_to_idx[cls]]).cuda())
            cur_batch = torch.tensor([class_to_idx[cls]]).size(0)
            cls_acc_1.updata(cur_cls_acc_1, cur_batch)


            cam = model.module.module.x_saliency[0][0].data.cpu() #1, 1, 28, 28


            if k< args.num_classes - 10:
                cam = cam + loc*args.adap_loc

            cam = normalize_map(np.array(cam),w,h)
            
            ##  制作bbox
            highlight = np.zeros(cam.shape)
            highlight[cam > args.threshold] = 1
            # max component
            all_labels = measure.label(highlight)
            highlight = np.zeros(highlight.shape)
            highlight[all_labels == count_max(all_labels.tolist())] = 1
            highlight = np.round(highlight * 255)
            highlight_big = cv2.resize(highlight, (w, h), interpolation=cv2.INTER_NEAREST)
            CAMs = copy.deepcopy(highlight_big)
            props = measure.regionprops(highlight_big.astype(int))

            if len(props) == 0:
                bbox = [0, 0, w, h]
            else:
                temp = props[0]['bbox']
                bbox = [temp[1], temp[0], temp[3], temp[2]]

            if TEN_CROP:
                img = ten_crop_aug(raw_img)
                img = img.to(0)
                

                adap_input, loc_adap_input = model(img, torch.tensor([class_to_idx[cls]]*10), inference=True)

                output1,_,_ = model(img, torch.tensor([class_to_idx[cls]]*10) ) #1, 100
                
                adap_output1 = adap(adap_input, adap=True) #1, 90, 14, 14

                adap_output1 = avg_pool(adap_output1).view(adap_output1.size(0), -1) #1, 90

                if k < args.num_classes -0:

                    front_exemplar=output1[:, :-10] + adap_output1*args.adap #64 50 14 14 + 64 50 14 14
                    back_exemplar =output1[:, -10:] # 64 10 14 14

                    output1 = torch.cat([front_exemplar,back_exemplar], dim=1)
                else:
                    front_exemplar=output1[:, :-10]  #64 50 14 14 + 64 50 14 14
                    back_exemplar =output1[:, -10:] # 64 10 14 14

                    output1 = torch.cat([front_exemplar,back_exemplar], dim=1)

                vgg16_out = output1
                vgg16_out = nn.Softmax()(vgg16_out)
                vgg16_out = torch.mean(vgg16_out,dim=0,keepdim=True)
                vgg16_out = torch.topk(vgg16_out, 5, 1)[1]


            else:
                img = cls_transform(raw_img)
                img = torch.unsqueeze(img, 0)
                img = img.to(0)
                vgg16_out,_,_ = model(img,[class_to_idx[cls]])
                vgg16_out = torch.topk(vgg16_out, 5, 1)[1]
            vgg16_out = to_data(vgg16_out)
            vgg16_out = torch.squeeze(vgg16_out)
            vgg16_out = vgg16_out.numpy()
            out = vgg16_out

        #handle resize and centercrop for gt_boxes

        gt_bbox_i = list(gt_boxes)
        raw_img_i = raw_img
        raw_img_i, gt_bbox_i = ResizedBBoxCrop((args.input_size,args.input_size))(raw_img, gt_bbox_i)
        raw_img_i, gt_bbox_i = CenterBBoxCrop((args.crop_size))(raw_img_i, gt_bbox_i)
        w, h = raw_img_i.size
        
        gt_box_num = len(gt_boxes) // 4
        for i in range(gt_box_num):
            gt_bbox_i[0+i*4] = gt_bbox_i[0+i*4] * w
            gt_bbox_i[2+i*4] = gt_bbox_i[2+i*4] * w
            gt_bbox_i[1+i*4] = gt_bbox_i[1+i*4] * h
            gt_bbox_i[3+i*4] = gt_bbox_i[3+i*4] * h

        gt_boxes = gt_bbox_i

        bbox[0] = bbox[0]    
        bbox[2] = bbox[2] 
        bbox[1] = bbox[1] 
        bbox[3] = bbox[3]  
        
        max_iou = -1
        
        for i in range(gt_box_num):
            gt_boxes = np.reshape(gt_boxes,(gt_box_num,4))
            iou = IoU(bbox, gt_boxes[i])
            if iou > max_iou:
                max_iou = iou
                max_box_num = i
        
        LocSet.append(max_iou)
        temp_loc_iou = max_iou
        if out[0] != class_to_idx[cls]:
            max_iou = 0

        IoUSet.append(max_iou)
        #cal top5 IoU
        max_iou = 0
        for i in range(5):
            if out[i] == class_to_idx[cls]:
                max_iou = temp_loc_iou
                count += 1
        IoUSetTop5.append(max_iou)
        
    cls_loc_acc = np.sum(np.array(IoUSet) > 0.5) / len(IoUSet)
    final_clsloc.extend(IoUSet)
    cls_loc_acc_top5 = np.sum(np.array(IoUSetTop5) > 0.5) / len(IoUSetTop5)
    final_clsloctop5.extend(IoUSetTop5)
    loc_acc = np.sum(np.array(LocSet) > 0.5) / len(LocSet)
    final_loc.extend(LocSet)
    print('{} cls-loc acc is {}, loc acc is {}, cls loc 5 is {}, cls top1 {}'.format(cls, cls_loc_acc, loc_acc, cls_loc_acc_top5, round(cls_acc_1.avg/100, 2)))
    accs.append(cls_loc_acc)
    accs_top5.append(cls_loc_acc_top5)
    loc_accs.append(loc_acc)
    if (k+1) %100==0:
        print(k)


print(accs)
print('Cls-Loc acc {}'.format(np.mean(accs)))
print('Cls-Loc acc Top 5 {}'.format(np.mean(accs_top5)))
print('GT Loc acc {}'.format(np.mean(loc_accs)))
print("classification top1 :", round(cls_acc_1.avg/100, 4))
print("classification top5 :", round(count/5000, 4))
print(args.threshold)
print("adap:",args.adap)
print("adap_loc:",args.adap_loc)