import argparse
from calendar import EPOCH
import os
from pickletools import uint8
from PIL import Image
import random
import numpy as np
import copy
from numpy import dot

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
torch.backends.cudnn.benchmark=True
import torchvision.transforms as transforms
from numpy.linalg import norm

from Model import *
from utils.loss import Loss
from utils.accuracy import *
from utils.optimizer import *
from utils.lr import *
from DataLoader import ImageDataset, ImageDataset10, ImageDataset100, ExemplarDataset, ImageDataset70
from models import *

from lib.util import moment_update, TransformTwice, weight_norm, weight_norm_dot, mixup_data, mixup_criterion, LabelSmoothingCrossEntropy
import time

seed = 2
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)

compute_means=True




class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser('argument for training')

        # training hyperparameters
        self.parser.add_argument('--start-epoch', type=int, default=1, help='number of training epochs')
        self.parser.add_argument('--epochs-sd', type=int, default=15, help='number of training epochs for self-distillation')
        self.parser.add_argument('--K', type=int, default=2000, help='memory budget')
        self.parser.add_argument('--save-freq', type=int, default=1, help='memory budget')
        
        # incremental learning    
        self.parser.add_argument('--new-classes', type=int, default=10, help='number of classes in new task')
        self.parser.add_argument('--start-classes', type=int, default=50, help='number of classes in old task')

        # dataset
        self.parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet100', 'imagenet', 'imagenet10', 'imagenet70'])

        # loss function
        self.parser.add_argument('--pow', type=float, default=0.66, help='hyperparameter of adaptive weight')
        self.parser.add_argument('--lamda', type=float, default=20, help='weighting of classification and distillation')
        self.parser.add_argument('--const-lamda', action='store_true', help='use constant lamda value, default: adaptive weighting')
        self.parser.add_argument('--w-cls', type=float, default=1.0, help='weightage of new classification loss')

        # kd loss
        self.parser.add_argument('--kd', action='store_true', help='use kd loss')
        self.parser.add_argument('--w-kd', type=float, default=1.0, help='weightage of knowledge distillation loss')
        self.parser.add_argument('--T', type=float, default=2, help='temperature scaling for KD')

        # self-distillation
        self.parser.add_argument('--num-sd', type=int, default=0, help='number of self-distillation generations')
        self.parser.add_argument('--sd-factor', type=float, default=5.0, help='weighting between classification and distillation')
########################################################################################################################################################
        ##  path
        self.parser.add_argument('--root', type=str, default='./media/data/imagenet100')
        self.parser.add_argument('--test_gt_path', type=str, default='val_gt_100.txt')
        self.parser.add_argument('--num_classes', type=int, default=100)
        self.parser.add_argument('--test_txt_path', type=str, default='val_list_100.txt')

        self.parser.add_argument('--save_path', type=str, default='logs/')
        self.parser.add_argument('--load_path', type=str, default='VGG.pth.tar')
        ##  image
        self.parser.add_argument('--crop_size', type=int, default=224)
        self.parser.add_argument('--resize_size', type=int, default=256) 
        ##  dataloader
        self.parser.add_argument('--num_workers', type=int, default=8)
        self.parser.add_argument('--nest', action='store_true')
        ##  train
        self.parser.add_argument('--batch_size', type=int, default=32*2)
        self.parser.add_argument('--epochs', type=int, default=9)
        self.parser.add_argument('--epochs_task', type=int, default=30)
        self.parser.add_argument('--pretrain', type=str, default='True')
        self.parser.add_argument('--phase', type=str, default='train')

        self.parser.add_argument('--delta', type=float, default=1.5)
        self.parser.add_argument('--fd', type=float, default=1.5)
        self.parser.add_argument('--fd2', type=float, default=1.5)
        self.parser.add_argument('--haalland', type=float, default=0.1)

        self.parser.add_argument('--lr', type=float, default=0.01)
        self.parser.add_argument('--lr_ft', type=float, default=0.001)

        self.parser.add_argument('--weight_decay', type=float, default=1e-4) # CCIL: 1e-4  BAS: 5e-4
        self.parser.add_argument('--power', type=float, default=0.9)
        self.parser.add_argument('--momentum', type=float, default=0.9)
        ##  model
        self.parser.add_argument('--arch', type=str, default='mobilenet')  ## choosen  [ vgg, resnet, inception, mobilenet, resnet18] 
        self.parser.add_argument('--adap', type=str, default='adaptation')
        ##  GPU'
        self.parser.add_argument('--gpu', type=str, default='0')      
        #adaptation
        self.parser.add_argument('--adaptation', type=str, default='True')   
      

    def parse(self):
        opt = self.parser.parse_args()
        opt.arch = opt.arch   
        return opt
avg_pool = nn.AvgPool2d(14)
avg_pool2 = nn.AvgPool2d(28)
def cos_sim(A, B):
                        return dot(A, B)/(norm(A)*norm(B))

def train(model, old_model, epoch, task, optimizer, lamda, train_loader, use_sd, checkPoint, now_task, adap, old_adap, optimizer_adap) :
    T = args.T 

    model.cuda()
    old_model.cuda()
    adap.cuda()
    old_adap.cuda()

    count = 0
    
    #criterion_ce = nn.CrossEntropyLoss(ignore_index=-1)
    
    if now_task > 1:

        lr = args.lr_ft
        epoch = args.epochs_task
        optimizer_adap = torch.optim.SGD(adap.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    else:
        lr = args.lr
        epoch = args.epochs
        optimizer_adap = torch.optim.SGD(adap.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    #optimizer = get_optimizer(model, args)
    
    if len(test_classes) // CLASS_NUM_IN_BATCH > 1:
        exemplar_set = ExemplarDataset(exemplar_sets)
        exemplar_loader = torch.utils.data.DataLoader(exemplar_set, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
        exemplar_loader_iter = iter(exemplar_loader) 
        old_model.eval()
        old_adap.eval()
    
    now = time.localtime()
    print ("%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))

    for epoch_index in range(args.start_epoch, epoch+1):
        #print(lr)

        dist_loss = 0.0
        sum_dist_loss = 0
        sum_cls_new_loss = 0
        sum_cls_old_loss = 0
        cam_loc_loss = 0

    
        model.eval()  
        model.module.freeze_weight() 
        old_model.eval()
        old_model.module.freeze_weight()
        adap.train()
        adap.module.unfreeze_weight()
        old_adap.eval()
        old_adap.module.freeze_weight()
  
            
        ##  accuracy
        cls_acc_1 = AverageMeter()
        cls_acc_2 = AverageMeter()
        loss_epoch_1 = AverageMeter()
        loss_epoch_2 = AverageMeter()
        loss_epoch_3 = AverageMeter()

        ## scheduler

        if now_task > 1:
           poly_lr_scheduler_task_adap(optimizer_adap, epoch_index)

        else:
           poly_lr_scheduler_adap(optimizer_adap, epoch_index)

        torch.cuda.synchronize()
        
        #for param_group in optimizer.param_groups:
        #    print('learning rate: {:.6f}'. format(param_group['lr']))

        for param_group_adap in optimizer_adap.param_groups:
            print('learning rate adap: {:.6f}'. format(param_group_adap['lr']))

        for batch_idx, (path, x, target) in enumerate(train_loader):

            optimizer.zero_grad()
            optimizer_adap.zero_grad()
            #print("i", i)

            # Classification Loss task 1 
            x, target = x.cuda(device=0), target.cuda(device=0)

            targets = target - len(test_classes) + CLASS_NUM_IN_BATCH
            #print(len(test_classes))
            with torch.no_grad():

                logits, loss_cls, loss_loc, x_new_data_new_model, new_input, target_new_x4, target_new_new_loc, target_new_new_loc_sig= model(x, targets, N=1, new=True, CLASS_NUM_IN_BATCH=CLASS_NUM_IN_BATCH) # [new new]
   
            loss_cls, loss_loc = loss_cls.mean(0), loss_loc.mean(0)
            targets = targets.long()

            loss = args.w_cls*loss_cls + loss_loc 
            sum_cls_new_loss += loss_cls.item()
            
            ##  count_accuracy
            cur_batch = targets.size(0)
            cur_cls_acc_1 = 100. * compute_cls_acc(logits[:, -CLASS_NUM_IN_BATCH:], targets) 
            cls_acc_1.updata(cur_cls_acc_1, cur_batch)
            loss_epoch_1.updata(loss_cls.data, 1)
            loss_epoch_2.updata(loss_loc.data, 1)

            # use fixed lamda value or adaptive weighting 
            if args.const_lamda:    
                factor = args.lamda
            else:
                factor = ((len(test_classes)/CLASS_NUM_IN_BATCH)**(args.pow))*args.lamda # args.pow = 0.66, args.lamda = 20

            # Distillation : task-2 onwards
            if len(test_classes) // CLASS_NUM_IN_BATCH > 1:

                # KD loss using new class data 
                if args.kd:
                    with torch.no_grad():
                        old_loss_cls, x_new_data_old_model, target_new_old_loc, x_input_o_current_data, x_input_o_current_data_loc, target_new_old_loc_sig= old_model(x, target, CCIL=True) # [old new]
                    #
                    

                    logits_dist = logits[:, :-CLASS_NUM_IN_BATCH] # (64,5)
                    x_new_data_new_model = x_new_data_new_model[:,:-CLASS_NUM_IN_BATCH, :, :]
                    T = args.T
                    dist_loss_new = nn.KLDivLoss()(F.log_softmax(logits_dist/T, dim=1), F.softmax(old_loss_cls/T, dim=1)) * (T*T)

                try:
                    batch_ex = next(exemplar_loader_iter)
                except StopIteration:
                    exemplar_loader_iter = iter(exemplar_loader)
                    batch_ex = next(exemplar_loader_iter)

                
                # Classification loss: old classes loss
                x_old, target_old = batch_ex
                x_old , target_old = x_old.cuda(device=0), target_old.cuda(device=0)
                with torch.no_grad():

                    logits_old, loss_cls_old, loss_loc_old= model(x_old, target_old, N=1) # [new old]

                loss_cls_old, loss_loc_old = loss_cls_old.mean(0), loss_loc_old.mean(0)
                target_old = target_old.long()
                loss += loss_cls_old + loss_loc_old 
                sum_cls_old_loss += loss_cls_old.item()

                ##  count_accuracy
                cur_batch1 = target_old.size(0)
                cur_cls_acc_2 = 100. * compute_cls_acc(logits_old, target_old) 
                cls_acc_2.updata(cur_cls_acc_2, cur_batch1)
                loss_epoch_1.updata(loss_cls_old.data, 1)
                loss_epoch_3.updata(loss_loc_old.data, 1)

                # KD loss using exemplars
                if args.kd: 
                    with torch.no_grad(): #0 50 60 70 80 90 
                        #new에 old dataset
                        dist_target_old, score_loc_old, old_feature, old_loc_feature, x_old_data_old_model, x_input_o, old_loc_adap, old_loc_adap_sig = old_model(x_old, target_old, N=1, feat=False, CCIL_old=True, cam=i) # [old old]

                        _, score_dist_loc_new, new_feature, new_loc_feature, x_old_data_new_model, x_input_n, new_loc_adap,  new_loc_adap_sig= model(x_old, target_old, CCIL_old=True, cam=i, N=1)  

                    logits_dist_old = logits_old[:, :-CLASS_NUM_IN_BATCH]
                    x_old_data_new_model = x_old_data_new_model[:, :-CLASS_NUM_IN_BATCH, :, :]


                    new_loc_adap = new_loc_adap[:, :-CLASS_NUM_IN_BATCH, :, :]
                    new_loc_adap_sig = new_loc_adap_sig[:, :-CLASS_NUM_IN_BATCH, :, :]
                    target_new_new_loc = target_new_new_loc[:, :-CLASS_NUM_IN_BATCH, :, :]
                    target_new_new_loc_sig = target_new_new_loc_sig[:, :-CLASS_NUM_IN_BATCH, :, :]
                    #new_loc_feature = new_loc_feature
                    
                    
                    dist_loss_old = nn.KLDivLoss()(F.log_softmax(logits_dist_old/T, dim=1), F.softmax(dist_target_old/T, dim=1)) * (T*T)  # best model
                 
                    l2loss = nn.MSELoss()
                    dist_loss_old_loc = l2loss(score_loc_old, score_dist_loc_new) #그냥 값으로 나옴
                   
                    
                    dist_feature = 1- F.cosine_similarity(old_feature, new_feature) #64 14 14 
                    dist_loc_feature = 1- F.cosine_similarity(old_loc_feature, new_loc_feature)   # 64 28 28   # new_loc_feature    64 512 28 28



                    dist_feature = avg_pool(dist_feature).view(dist_feature.size(0), -1)
                    dist_loc_feature = avg_pool2(dist_loc_feature).view(dist_loc_feature.size(0), -1) #64
                    dist_feature = torch.mean(dist_feature)
                   
                    dist_loc_feature = torch.mean(dist_loc_feature) #1
                    #print(dist_loc_feature)
               
                    
                    dist_loss =  dist_loss_old + dist_loss_new + dist_feature * args.fd  + dist_loc_feature * args.fd2 + (dist_loss_old_loc * args.delta) 
                    sum_dist_loss += dist_loss.item()
                    cam_loc_loss += dist_loss_old_loc.item()
                    loss += factor*args.w_kd*dist_loss


                    if len(test_classes) // CLASS_NUM_IN_BATCH > 1 and args.kd and now_task ==2 :  #70일때 69

                        x1 = adap(x_input_n, target, adap=True) #new model에 target old
                        x1 = x_old_data_new_model +x1

                        x1_loc = adap(new_loc_feature, target) #64 50 28 28
                        x1_loc = x1_loc + new_loc_adap_sig

                        x2 = adap(new_input, target, adap=True) #new model에 target new
                        x2 = x2 + x_new_data_new_model

                        x2_loc = adap(target_new_x4, target) #64 50 28 28
                        x2_loc = x2_loc + target_new_new_loc_sig


                        adap_loss = l2loss(x_new_data_old_model, x2)*args.haalland + l2loss(x1, x_old_data_old_model)*args.haalland+ \
                            l2loss(x1_loc, old_loc_adap_sig)*args.haalland + l2loss(x2_loc, target_new_old_loc_sig)*args.haalland
                        
                        #print("Train adaptation:::   adaptation loss:", adap_loss)
                        #
                        # if (batch_idx + 1) % checkPoint == 0 or (batch_idx+1) == 1:
                        #     print("####################")

                        #     print(l2loss(x_new_data_old_model, x2))
                        #     print(l2loss(x1, x_old_data_old_model))
                        #     print(l2loss(x1_loc, old_loc_adap))
                        #     print(l2loss(x2_loc, target_new_old_loc))
                        #     print(adap_loss)

                    elif len(test_classes) // CLASS_NUM_IN_BATCH > 1 and args.kd and now_task >2 :  #70일때 69 -10

                        #exemplar set
                        x1 = adap(x_input_n, target_old, adap=True) # 64 60 14 14
                        x1 = x_old_data_new_model +x1  # p3 +a3 # 64 60 14 14 + 64 60 14 14
                        x1_a =  old_adap(x_input_o, target, adap=True) #a2 # 64 50 14 14
                        
                        front_exemplar=x_old_data_old_model[:, :-CLASS_NUM_IN_BATCH, :, :] + x1_a #64 50 14 14 + 64 50 14 14
                        back_exemplar =x_old_data_old_model[:, -CLASS_NUM_IN_BATCH:, :, :] # 64 10 14 14
                        combine_exemplar = torch.cat([front_exemplar,back_exemplar], dim=1)  # 64 60 14 14

                        x1_loc = adap(new_loc_feature, target_old) # 64 60 14 14
                        #new_loc_adap = torch.log(new_loc_adap) - torch.log(1-new_loc_adap)
                        x1_loc = new_loc_adap_sig + x1_loc  # p3 +a3 # 64 60 14 14 + 64 60 14 14
                        
                        x1_a_loc =  old_adap(old_loc_feature, target) #a2 # 64 50 14 14
                        
                        #old_loc_adap = torch.log(old_loc_adap) - torch.log(1-old_loc_adap)
                        front_exemplar_loc=old_loc_adap_sig[:, :-CLASS_NUM_IN_BATCH, :, :] + x1_a_loc #64 50 14 14 + 64 50 14 14
                        back_exemplar_loc =old_loc_adap_sig[:, -CLASS_NUM_IN_BATCH:, :, :] # 64 10 14 14
                        combine_exemplar_loc = torch.cat([front_exemplar_loc,back_exemplar_loc], dim=1)  # 64 60 14 14


                        #new data
                        x2 = adap(new_input, target, adap=True) # 64 60 14 14
                        x2 = x2 + x_new_data_new_model # p3 +a3 # 64 60 14 14 + 64 60 14 14
                        x2_a = old_adap(x_input_o_current_data, target, adap=True) #a2 # 64 50 14 14

                        front=x_new_data_old_model[:, :-CLASS_NUM_IN_BATCH, :, :] + x2_a #64 50 14 14 + 64 50 14 14
                        back =x_new_data_old_model[:, -CLASS_NUM_IN_BATCH:, :, :] # 64 10 14 14
                        combine = torch.cat([front,back], dim=1)  # 64 60 14 14


                        x2_loc = adap(target_new_x4, target_old) # 64 60 14 14
                        #target_new_new_loc = torch.log(target_new_new_loc) - torch.log(1-target_new_new_loc)
                        x2_loc = target_new_new_loc_sig + x2_loc  # p3 +a3 # 64 60 14 14 + 64 60 14 14
                        x2_a_loc =  old_adap(x_input_o_current_data_loc, target) #a2 # 64 50 14 14
                        
                        #target_new_old_loc = torch.log(target_new_old_loc) - torch.log(1-target_new_old_loc)
                        front_loc=target_new_old_loc_sig[:, :-CLASS_NUM_IN_BATCH, :, :] + x2_a_loc #64 50 14 14 + 64 50 14 14
                        back_loc =target_new_old_loc_sig[:, -CLASS_NUM_IN_BATCH:, :, :] # 64 10 14 14
                        combine_loc = torch.cat([front_loc, back_loc], dim=1)  # 64 60 14 14

                        #target_new_new_loc = torch.log(target_new_new_loc) - torch.log(1-target_new_new_loc)
                        #target_new_new_loc = torch.log(target_new_new_loc) - torch.log(1-target_new_new_loc)

                        
                        #[:, :-CLASS_NUM_IN_BATCH, :, :] #60 + 50, 60
                        adap_loss = l2loss(combine, x2)*args.haalland + l2loss(combine_exemplar, x1)*args.haalland \
                            + l2loss(combine_loc, x2_loc)*args.haalland + l2loss(combine_exemplar_loc, x1_loc)*args.haalland
                        

                        
                        
                        #p l2loss(p2+a2 , p3+a3)
                        #print(adap_loss)


                        #print("Train adaptation:::   adaptation loss:", adap_loss)


            
            if len(test_classes) // CLASS_NUM_IN_BATCH > 1:
                
                adap_loss.requires_grad_(True)  
                adap_loss += adap_loss + 1e-6
                adap_loss.backward()
                optimizer_adap.step()
                #print(adap_loss)
                if (batch_idx + 1) % checkPoint == 0 or (batch_idx+1) == 1:
                    print(adap_loss)


            if (batch_idx + 1) % checkPoint == 0 or (batch_idx+1) == 1:
                #print(type(x_input_o_current_data_loc))
                print('==>>> epoch: [{}/{}], batch index: [{}/{}], kd_loss: {:3f}, cls_new_loss: {:.3f}, cls_old_loss: {:.3f}, loss_cls:{:.3f}, loss_bas:{:.3f}, loss_bas_old:{:.3f}, epoch_acc_1:{:.2f}, epoch_acc_2:{:.2f}, dist_cam:{:.2f}'.
                        format(epoch_index, epoch, batch_idx + 1, len(MyDataLoader), sum_dist_loss/(batch_idx+1), sum_cls_new_loss/(batch_idx+1),  sum_cls_old_loss/(batch_idx+1), loss_epoch_1.avg, loss_epoch_2.avg, loss_epoch_3.avg, cls_acc_1.avg, cls_acc_2.avg, cam_loc_loss))
        

        torch.save(adap, os.path.join(args.save_path, args.adap + '_' + str(task) + '_' + str(epoch) +'_'+ 'final' +'.pth.tar'), _use_new_zipfile_serialization=False )
            
       
        now = time.localtime()
        print ("%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))

        

def icarl_reduce_exemplar_sets(m, exemplar_sets):
        exemplar_sets =exemplar_sets

        for y, P_y in enumerate(exemplar_sets):
            exemplar_sets[y] = P_y[:m]
        print("##########################################")
        print ('exemplar_sets len: ', len(exemplar_sets))
        print("##########################################")
        return exemplar_sets
        
        


def sorting(model, images, m, transform, all_sets):
    model.eval()
    # Compute and cache features for each example
    features = []
    #path1 = []
    exemplar_sets = all_sets #두번째 task끝나고는 33장씩 들어있는 0~49의 클래스
    print(len(exemplar_sets))
    
    
    with torch.no_grad():
        
        for img in images: # each class image feature 
            x = Variable(transform(Image.fromarray(img))).cuda() # array to image
            x=x.unsqueeze(0)
        
            feat = model.cuda().forward(x, feat=True).data.cpu().numpy()
            feat = feat / np.linalg.norm(feat) # Normalize

            features.append(feat[0])

        features = np.array(features)
        print("features check",len(features[0]))
    
        class_mean = np.mean(features, axis=0)
        class_mean = class_mean / np.linalg.norm(class_mean) # Normalize 200704 1024*14*14

        exemplar_set = []
        exemplar_features = [] # list of Variables of shape (feature_size,)
        exemplar_dist = []
        mu = class_mean

        mu_p = features
        mu_p = mu_p / np.linalg.norm(mu_p)
        dist = np.sqrt(np.sum((mu - mu_p) ** 2, axis=1))
        abc = dist.argsort()

        for k in range(images.shape[0]):

            exemplar_set.append(images[k])

        abc = np.array(abc)
        exemplar_set = np.array(exemplar_set)

        abc = abc[:m]
        exemplar_set = exemplar_set[abc]
        exemplar_sets.append(np.array(exemplar_set))

    print ('sorting shape: ', len(exemplar_set))
    print ('exemplar_sets len: ', len(exemplar_sets))
    return exemplar_sets

#Construct an exemplar set for image set
def icarl_construct_exemplar_set(model, images, m, transform, exemplar_sets):
    
    #exemplar_sets = []
    exemplar_sets = exemplar_sets
    model.eval()
    with torch.no_grad():
       # print(images.shape[0])
        tmp_set = []
        for k in range(int(m)):
            i = np.random.randint(0, images.shape[0])
            tmp_set.append(images[i])
        tmp_set = np.array(tmp_set)

        exemplar_sets.append(np.array(tmp_set))
    print ('random exemplar set shape: ', len(tmp_set))
    print ('exemplar_sets len: ', len(exemplar_sets))
    return exemplar_sets



args = opts().parse()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
lr = args.lr
epoch = args.epochs
now_task=1

if __name__ == '__main__':
    print (args)
    exemplar_sets = []
    
    all_sets = []
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    
    #  parameters
    if args.dataset == 'imagenet100':
        TOTAL_CLASS_NUM = 100
    elif args.dataset == 'imagenet':
        TOTAL_CLASS_NUM = 1000
    elif args.dataset == 'imagenet10':
        TOTAL_CLASS_NUM = 10
    elif args.dataset == 'imagenet70':
        TOTAL_CLASS_NUM = 70
        
    CLASS_NUM_IN_BATCH = args.start_classes
    TOTAL_CLASS_BATCH_NUM = TOTAL_CLASS_NUM // CLASS_NUM_IN_BATCH
    T = args.T

    exemplar_means = []    
    compute_means = True

    # test-time augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    class_index = [i for i in range(0, TOTAL_CLASS_NUM)] # imagenet100 [0 ~ 100], imagenet1000 [0 ~ 1000]
    net = torch.load(os.path.join(args.save_path, args.arch + '_' + '50' + '_' + '12' + '_'+ 'final' +'.pth.tar')) 
   # net = eval(args.arch).model(args, num_classes=CLASS_NUM_IN_BATCH, pretrained=True) # CLASS_NUM: 50 or 500           
   # net = nn.DataParallel(net, device_ids=[0])
    net.cuda(device=0)
    net = net
    net.eval()

    adap = eval(args.adap).model(args, num_classes=CLASS_NUM_IN_BATCH, pretrained=False) # CLASS_NUM: 50 or 500           
    adap = nn.DataParallel(adap, device_ids=[0])
    adap.cuda(device=0)
    adap = adap
    #print(adap)
    adap.train()


    optimizer_adap = get_optimizer(adap, args)

    old_net = copy.deepcopy(net)
    old_net.cuda()

    old_adap = copy.deepcopy(adap)
    old_adap.cuda()
   
    cls_list = [0] + [a for a in range(args.start_classes, TOTAL_CLASS_NUM, args.new_classes)] # imagenet100 = [0, 50, 60, 70, 80, 90]. imagenet = [0, 500, 600, 700, 800, 900]

    print(cls_list)

    for i in cls_list:
        #i=50
        if i == args.start_classes:
            CLASS_NUM_IN_BATCH = args.new_classes # CLASS_NUM_IN_BATCH = [50, 10, 10, 10, 10, 10]

        print("==> Current Class: ", class_index[i:i+CLASS_NUM_IN_BATCH])
        print('==> Building model..')
  
        if i == args.start_classes: #task2
            net = net = torch.load(os.path.join(args.save_path, args.arch + '_' + '60' + '_' + '70' + '_'+ 'final' +'.pth.tar'))  # new_dim: 50 + 10 = 60

            adap = eval(args.adap).model(args, num_classes=i) # CLASS_NUM: 50 or 500 pretrain 상관x
            adap = nn.DataParallel(adap, device_ids=[0])
            adap.cuda(device=0)
            adap = adap
            adap.train()
            optimizer_adap = get_optimizer(adap, args)
            

        if i > args.start_classes:
            net = torch.load(os.path.join(args.save_path, args.arch + '_' + str(i+CLASS_NUM_IN_BATCH) + '_' + '70' + '_'+ 'final' +'.pth.tar'))
            
            adap = eval(args.adap).model(args, num_classes=i) # CLASS_NUM: 50 or 500           
            adap = nn.DataParallel(adap, device_ids=[0])
            adap.cuda(device=0)
            adap = adap
            adap.train()
            optimizer_adap = get_optimizer(adap, args)
            

        print("current net output dim:", net.module.get_output_dim())
        print("old net output dim:", old_net.module.get_output_dim())

        if args.dataset == 'imagenet100':
            MyData = ImageDataset100(args, i=i)
        elif args.dataset == 'imagenet':
            MyData = ImageDataset(args, i=i)
        elif args.dataset == 'imagenet10':
            MyData = ImageDataset10(args, i=i)
        elif args.dataset == 'imagenet70':
            MyData = ImageDataset70(args, i=i)

        MyDataLoader = torch.utils.data.DataLoader(dataset=MyData, batch_size=args.batch_size,shuffle=True, num_workers= args.num_workers,pin_memory=True)
   
        train_classes = class_index[i:i+CLASS_NUM_IN_BATCH]
        test_classes = class_index[:i+CLASS_NUM_IN_BATCH]

        print ("train_classes:",train_classes)
        print ("test_classes:",test_classes)    

        m = args.K // (i+CLASS_NUM_IN_BATCH) # 몫 [40, 33, 28, 25, 22, 20]
        #print(net)
        print("task:", now_task-1)
       
        if i!=0:
            
            exemplar_sets = icarl_reduce_exemplar_sets(m, exemplar_sets = exemplar_sets) 
            all_sets = copy.deepcopy(exemplar_sets)
            print( all_sets is exemplar_sets)
            print("@@@@@@@@@@@@@@@@@@@@@@")
            print("all_sets len",len(all_sets))
            print("@@@@@@@@@@@@@@@@@@@@@@")


        for y in range(i, i+CLASS_NUM_IN_BATCH):
            now = time.localtime()
            print ("%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))
            print ("Randomly sampling exemplar set for class-%d..." %(class_index[y]))
            images = MyData.get_image_class(y)
            exemplar_sets = icarl_construct_exemplar_set(net, images, m, transform_test, exemplar_sets = exemplar_sets)
            print ("Done")

        net.train()
        

        train(model=net, old_model=old_net, epoch=args.epochs, task= i+CLASS_NUM_IN_BATCH, optimizer=optimizer_adap, lamda=args.lamda, train_loader=MyDataLoader
        , use_sd=False, checkPoint=50, now_task=now_task, adap=adap, old_adap=old_adap, optimizer_adap = optimizer_adap)
        
        if now_task > 1: # 1 2 3 4 5 6
            epoch = args.epochs_task
        else:
            epoch = args.epochs
        
        if now_task == 1:
            old_net = torch.load(os.path.join(args.save_path, args.arch + '_' + '50' + '_' + '12' + '_'+ 'final' +'.pth.tar'))

        else:

            old_net = torch.load(os.path.join(args.save_path, args.arch + '_' + str(i+CLASS_NUM_IN_BATCH) + '_' + '70' + '_'+ 'final' +'.pth.tar'))
        old_adap = torch.load(os.path.join(args.save_path, args.adap + '_' + str(i+CLASS_NUM_IN_BATCH) + '_' + str(epoch) + '_'+ 'final' +'.pth.tar'))
        # print(os.path.join(args.save_path, args.arch + '_' + str(i+CLASS_NUM_IN_BATCH) + '_' + str(args.epochs) +'_'+ 'final' +'.pth.tar'))
        old_net.cuda()
        old_adap.cuda()



        print("aaaa",len(all_sets))
        exemplar_sets.clear()
        print("bbbb",len(all_sets))
        
        exemplar_sets = []
        

        if now_task > 5:  #5
            pass
        else:
            for y in range(i, i+CLASS_NUM_IN_BATCH):
                now = time.localtime()
                print ("%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))
                print ("Sorting exemplar set for class-%d..." %(class_index[y]))
                images = MyData.get_image_class(y)

                exemplar_sets = sorting(net, images, m, transform_test, all_sets = all_sets)
                print ("Done")

        

        now_task += 1