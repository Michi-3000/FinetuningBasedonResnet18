import pickle
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from matplotlib import cm as CM
from matplotlib import axes
from torch.autograd import Variable
from torch import optim
from PIL import Image
from os import path
from functools import partial
import cv2


RGB_mean = [0.5453, 0.5283, 0.5022]
RGB_std = [0.2422, 0.2392, 0.2406]

def resnet18_101(**kwargs):
    net = models.resnet18(**kwargs)
    for params in net.parameters():
        params.requires_grad = False
    featureSize = net.fc.in_features
    #net = net.view(-1, featureSize)
    net.fc = torch.nn.Linear(featureSize, 101) #改变全连接层
    
    return net

def my_forward(model, x):
    mo = nn.Sequential(*list(model.children())[:-1])
    feature = mo(x)
    feature = feature.view(x.size(0), -1)
    output= model.fc(feature)
    return feature, output

def hook(module, input, output, feature_blob):
    feature_blob.append(output.data.numpy())

def compute_cam(activation, softmax_weight, class_ids):
    '''
    activation:  最后一个conv层输出的结果  1, 512, 7, 7
    softmax_weight: 全连接的权重 101*512
    class_ids: 按照概率从大到小排序后的概率的下标
    '''
    b, c, h, w = activation.shape #b: batchsize c: channel 通道数 h*w: 图像大小
    cams = []
    for idx in class_ids:
        activation = activation.reshape(c, h * w) #512*49
        # dot 计算两个张量的点乘 (内积)
        cam = softmax_weight[idx].dot(activation) #权重乘特征图
        cam = cam.reshape(h, w)
        cam =  (cam - cam.min()) / (cam.max() - cam.min()) # 归一化
        cam = np.uint8(255 * cam) # 转化到0-255之间
        cams.append(cv2.resize(cam, (224, 224))) # 从[7,7]resize到[224,224]
    return cams

#L_path = './101_Categories/test/yin_yang/image_0045.jpg' #12
#L_path = './101_Categories/test/starfish/image_0025.jpg' #3
#L_path = './101_Categories/test/camera/image_0028.jpg'
#L_path = './101_Categories/test/chair/image_0046.jpg' #49
#L_path = './101_Categories/test/dollar_bill/image_0046.jpg' #45
#L_path = './101_Categories/test/crocodile/image_0014.jpg'
#L_path = './101_Categories/test/joshua_tree/image_0003.jpg'
L_path = './101_Categories/test/accordion/image_0003.jpg'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(RGB_mean, RGB_std)
    ]
)

def get_test_img():
    img_origin = Image.open(L_path) #PIL读入是RGB通道的图片
    img = transform(img_origin)
    return img_origin, img

def visualize_cam():
    net = resnet18_101(pretrained=True)
    net.load_state_dict(torch.load('params.pkl', map_location=lambda storage, loc: storage)) # 加载模型和训练好的参数
    net.eval()

    feature_blob = [] #存储的为最后一个conv层输出的结果
    net.layer4.register_forward_hook(partial(hook, feature_blob = feature_blob))   # 为最后一个conv层添加一个hook

    params = list(net.parameters()) #参数
    softmax_weight = np.squeeze(params[-2].data.numpy()) # 得到fc层的权重 #torch.Size([101*512])（若axis为空，则删除所有单维度的条目）
    
    img_origin, img = get_test_img() #torch.Size([3, 224, 224])
    output = net(Variable(img).unsqueeze(0)) #增加batchsize维度
    output = F.softmax(output).data.squeeze() #一行一行做归一化
    
    # 按照概率进行从大到小排序
    output, idx = output.sort(0, descending = True)

    id_pre = []
    id_pre.append(idx[0])
    cams = compute_cam(feature_blob[0], softmax_weight, id_pre) #feature_blob[0]为最后一个conv层输出的结果 (1, 512, 7, 7)
    w, h = img_origin.size #原图的尺寸
    #画出CAM
    CAM = cv2.applyColorMap(cv2.resize(cams[0], (w, h)), cv2.COLORMAP_JET)
    cv2.imwrite('out_CAM.jpg', CAM)
    # 将CAM和原图叠加在一起
    CAM = cv2.imread('out_CAM.jpg')
    img_origin = cv2.imread(L_path)
    result = CAM * 0.6 + img_origin * 0.5
    cv2.imwrite('out_ans.jpg', result)

if __name__ == '__main__':
    visualize_cam()