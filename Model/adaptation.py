import torch
import torch.nn as nn 
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np 
import cv2
from skimage import measure
from utils.func import *
import torchvision.models as models
from Model.modified_linear_mobile import *


class Model(nn.Module):
    def __init__(self, args, num_classes=1000):
        super(Model, self).__init__()

        self.adaptation = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1), #1024 14 14
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, num_classes, kernel_size=1, padding=0),
            #nn.ReLU(inplace=True), 
            #nn.Tanh(),
        )
        self.adaptation_loc = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_classes, kernel_size=3, padding=1),
            #nn.Tanh(),
        )
        self._initialize_weights()


    def forward(self, x, label=None, N=1, feat=False, CCIL=False, CCIL_old=False, new=False, loc=False, CLASS_NUM_IN_BATCH=500, cam=1, fd=False, adap=False):
        if adap:
            x = self.adaptation(x)
            
        else:
            x = self.adaptation_loc(x)
        return x

    def normalize_atten_maps(self, atten_maps):
        atten_shape = atten_maps.size()

        #--------------------------
        batch_mins, _ = torch.min(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        batch_maxs, _ = torch.max(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        atten_normed = torch.div(atten_maps.view(atten_shape[0:-2] + (-1,))-batch_mins,
                                 batch_maxs - batch_mins + 1e-10)
        atten_normed = atten_normed.view(atten_shape)

        return atten_normed

            
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
    def weight_deepcopy(self):
        for i in range(len(self.classifier_cls)):
            if 'Conv' in str(self.classifier_cls[i]) or 'BatchNorm2d' in str(self.classifier_cls[i]):
                self.classifier_cls_copy[i].weight.data = self.classifier_cls[i].weight.clone().detach()
                self.classifier_cls_copy[i].bias.data = self.classifier_cls[i].bias.clone().detach()
        for i in range(len(self.model[-2:])):
            for j in range(len(self.model[-2:][i])):
                if 'Conv' in str(self.model[-2:][i][j]) or 'BatchNorm2d' in str(self.model[-2:][i][j]):
                    self.erase_branch[i][j].weight.data = self.model[-2:][i][j].weight.clone().detach()
                if 'BatchNorm2d' in str(self.model[-2:][i][j]):
                    self.erase_branch[i][j].bias.data = self.model[-2:][i][j].bias.clone().detach()

    def freeze_weight(self):
            for param in self.parameters():
                param.requires_grad = False
                
    def unfreeze_weight(self):
            for param in self.parameters():
                param.requires_grad = True

    def get_output_dim(self):
        return self.classifier_cls[3].out_channels

    def change_output_dim(self, new_dim, second_iter=False):

        if second_iter:
            in_channels1 = self.classifier_cls[3].in_channels
            in_channels2 = self.classifier_loc[0].in_channels

            out_channels1 = self.classifier_cls[3].fc1.out_channels
            out_channels2 = self.classifier_cls[3].fc2.out_channels

            out_channels3 = self.classifier_loc[0].fc1.out_channels
            out_channels4 = self.classifier_loc[0].fc2.out_channels

            new_fc = SplitCosineLinear(in_channels1, out_channels1+out_channels2, out_channels2)
            new_loc = change_loc(in_channels2, out_channels3+out_channels4, out_channels4)

            new_fc.fc1.weight.data[:out_channels1] = self.classifier_cls[3].fc1.weight.data
            new_fc.fc1.weight.data[out_channels1:] = self.classifier_cls[3].fc2.weight.data

            new_loc.fc1.weight.data[:out_channels3] = self.classifier_loc[0].fc1.weight.data
            new_loc.fc1.weight.data[out_channels3:] = self.classifier_loc[0].fc2.weight.data
            new_fc.sigma.data = self.classifier_cls[3].sigma.data
            #new_loc.sigma.data = self.classifier_loc[0].sigma.data
            self.classifier_cls[3] = new_fc
            #print(self.classifier_cls[4].out_channels)
            self.classifier_loc[0] = new_loc
            new_out_channels = new_dim
            self.n_classes = new_out_channels
            self.classifier_cls_copy = copy.deepcopy(self.classifier_cls) 

        else:
            in_channels1 = self.classifier_cls[3].in_channels
            in_channels2 = self.classifier_loc[0].in_channels

            out_channels1 = self.classifier_cls[3].out_channels
            out_channels2 = self.classifier_loc[0].out_channels

            new_out_channels = new_dim   # 50

            num_new_classes1 = new_dim - out_channels1
            num_new_classes2 = new_dim - out_channels2

            new_fc = SplitCosineLinear(in_channels1, out_channels1, num_new_classes1)
            new_fc.fc1.weight.data = self.classifier_cls[3].weight.data # [new_fc.fc1.in_features = 512, new_fc.fc1.out_features = 50 ], [new_fc.fc2.in_features = 512, new_fc.fc1.out_features = 10]
            new_loc = change_loc(in_channels2, out_channels2, num_new_classes2)
            new_loc.fc1.weight.data = self.classifier_loc[0].weight.data

            new_fc.sigma.data = self.classifier_cls[3].sigma.data
            #new_loc.sigma.data = self.classifier_loc[0].sigma.data

            self.classifier_cls[3] = new_fc
            self.classifier_loc[0] = new_loc
            self.n_classes = new_out_channels
            self.classifier_cls_copy = copy.deepcopy(self.classifier_cls) 

def model(args, num_classes, pretrained=True):
    model = Model(args, num_classes)
    
    # if pretrained:
    #     pretrained_dict = torch.load('mobilenet_v1_with_relu_69_5.pth')
    #     model_dict = model.state_dict() 
    #     model_conv_name = []

    #     for i, (k, v) in enumerate(model_dict.items()):
    #         if 'tracked' in k[-7:]:
    #             continue
    #         model_conv_name.append(k)
    #     for i, (k, v) in enumerate(pretrained_dict.items()):
    #         model_dict[model_conv_name[i]] = v 
    #     model.load_state_dict(model_dict)
    #     print("pretrained weight load complete..")
    return model