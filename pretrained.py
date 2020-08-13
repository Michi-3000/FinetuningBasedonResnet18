import torchvision.models as models
import torchvision.transforms as transforms
import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.autograd import Variable
from os import path
import ssl # 全局取消证书验证

root = "./101_Categories/"

RGB_mean = [0.5453, 0.5283, 0.5022]
RGB_std = [0.2422, 0.2392, 0.2406]

def build_data(data_set,batch_size=20):   
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(RGB_mean, RGB_std)
    ]) 

    data_dir = path.join('./101_Categories', data_set)
    dataset = torchvision.datasets.ImageFolder(data_dir, transform=data_transform)
    dataloadder = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, dataloadder

def resnet18_101(**kwargs):
    net = models.resnet18(**kwargs)
    for params in net.parameters():
        params.requires_grad = False
    featureSize = net.fc.in_features
    net.fc = torch.nn.Linear(featureSize, 101) #改变全连接层
    
    return net

def resnet():
   print("Reading train data...")
   train_data, train_loader = build_data('train')
   print("Reading validation data...")
   val_data, val_loader = build_data('val')
   print("Reading test data...")
   test_data, test_loader = build_data('test')
   # GPU or CPU
   if torch.cuda.is_available():
       is_cuda = True
       print("work on GPU")
   else:
       is_cuda = False
       print("work on CPU")

   print("Setup Net...")
   if is_cuda:
       model = resnet18_101(pretrained=True).cuda()
   else:
       model = resnet18_101(pretrained=True)

   # 打印网络结构
   print(model)
   LR=0.00005 
   fc_params = list(map(id, model.fc.parameters()))
   base_params = filter(lambda p: id(p) not in fc_params,model.parameters())
   optimizer = torch.optim.Adam([
            {'params': base_params},
            {'params': model.fc.parameters(), 'lr': LR * 10}], LR) 

   loss_func = torch.nn.CrossEntropyLoss()

   for epoch in range(50):  # 训练100批次
       print('epoch {}'.format(epoch + 1))
       # 训练
       for p in optimizer.param_groups:
          p['lr'] *= 0.98
       train_loss = 0.
       train_acc = 0.
       cnt = 0
       total = 0
       for batch_x, batch_y in train_loader:  # 特征 标号
           if is_cuda:
              batch_x = batch_x.cuda()
              batch_y = batch_y.cuda()
           #print(batch_y)
           out = model(batch_x)  # batch_x通过网络的结果是out
           loss = loss_func(out, batch_y)  # 网络结果out和实际batch_y对比的得到损失
           train_loss += loss.item()  # 累加训练损失
           if is_cuda:
              pred = torch.max(out, 1)[1].cuda()  # 返回最大值的索引
           else:
              pred = torch.max(out, 1)[1]  # 返回最大值的索引
           train_correct = (pred == batch_y).sum()  # 多少个预测为正确的
           total += len(batch_x)
           train_acc += train_correct.item()  # 累加训练正确的数量
           print('[epoch:%d, iter:%d] Loss: %.6f | Acc: %.6f '
                          % (epoch + 1, (cnt + 1), train_loss / total, train_acc / total))
           optimizer.zero_grad()  # 清除所有优化的grad
           loss.backward()  # 误差反向传递
           optimizer.step()  # 单次优化
           cnt += 1
       print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(train_data)), train_acc / (len(train_data))))

       # 验证
       model.eval()
       eval_loss = 0.
       eval_acc = 0.
       cnt = 0
       total = 0
       for batch_x, batch_y in val_loader:  # 特征 标号
           if is_cuda:
               batch_x, batch_y = Variable(batch_x).cuda(), Variable(batch_y).cuda()
           else:
               batch_x, batch_y = Variable(batch_x), Variable(batch_y)
           out = model(batch_x)
           loss = loss_func(out, batch_y)
           eval_loss += loss.item()
           if is_cuda:
              pred = torch.max(out, 1)[1].cuda()  # 返回最大值的索引
           else:
              pred = torch.max(out, 1)[1]
           num_correct = (pred == batch_y).sum()
           eval_acc += num_correct.item()
           total += len(batch_x)
           print('[vepoch:%d, iter:%d] Loss: %.6f | Acc: %.6f '
                % (epoch + 1, (cnt + 1), eval_loss / total, eval_acc / total))
           cnt += 1
       print('validation Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(val_data)), eval_acc / (len(val_data))))


  # 测试
   model.eval()
   eval_loss = 0.
   eval_acc = 0.
   for batch_x, batch_y in test_loader:  # 特征 标号
       if is_cuda:
          batch_x, batch_y = Variable(batch_x).cuda(), Variable(batch_y).cuda()
       else:
          batch_x, batch_y = Variable(batch_x), Variable(batch_y)
       out = model(batch_x)
       loss = loss_func(out, batch_y)
       eval_loss += loss.item()
       if is_cuda:
          pred = torch.max(out, 1)[1].cuda()
       else:
          pred = torch.max(out, 1)[1]
       num_correct = (pred == batch_y).sum()
       eval_acc += num_correct.item()
   print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_data)), eval_acc / (len(test_data))))
   #print('Test Acc: {:.6f}'.format(eval_acc / (len(test_data))))
if __name__ == '__main__':
    ssl._create_default_https_context = ssl._create_unverified_context
    resnet()