import torchvision.models as models
import torchvision.transforms as transforms
import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.autograd import Variable
from os import path
import ssl
from torch.nn import init

root = "./101_Categories/"

RGB_mean = [0.5453, 0.5283, 0.5022]
RGB_std = [0.2422, 0.2392, 0.2406]


def initNetParams(net):
    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            init.xavier_uniform(m.weight)
            if m.bias is not None:
                init.constant(m.bias, 0)
        elif isinstance(m, torch.nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant(m.bias, 0)


def build_data(data_set,batch_size=20):
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), #把灰度范围从0-255变换到0-1之间
        transforms.Normalize(RGB_mean, RGB_std) #image=(image-mean)/std 
    ])

    data_dir = path.join('./101_Categories', data_set)
    dataset = torchvision.datasets.ImageFolder(data_dir, transform=data_transform)
    dataloadder = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, dataloadder

class ResBlock(torch.nn.Module):
  def __init__(self, IN, OUT, S, change_channels=False):
    super(ResBlock, self).__init__()
    self.change_channels = change_channels
    self.resblock = torch.nn.Sequential(
      torch.nn.Conv2d(in_channels=IN, out_channels=OUT, kernel_size=3, stride=S, padding=1, bias=False),
      torch.nn.BatchNorm2d(OUT),
      torch.nn.ReLU(inplace=True),
      torch.nn.Conv2d(in_channels=OUT, out_channels=OUT, kernel_size=3, stride=1, padding=1, bias=False),
      torch.nn.BatchNorm2d(OUT)
      )
    if self.change_channels:
      self.Res= torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=IN, out_channels=OUT, kernel_size=1, stride=S, bias=False),
        torch.nn.BatchNorm2d(OUT)
        )
    self.relu = torch.nn.ReLU(inplace=True)

  def forward(self, x):
    y = self.resblock(x)
    res = x
    if self.change_channels:
      res = self.Res(x)
    y += res
    y = self.relu(y)
    return y

def copy_res(IN, OUT, S, cnt):
  net = []
  net.append(ResBlock(IN, OUT, S, change_channels=True))
  for i in range(1, cnt):
    net.append(ResBlock(OUT, OUT, 1, change_channels=False))
  return torch.nn.Sequential(*net)


# 网络结构
class Net(torch.nn.Module):
   def __init__(self):
       super(Net, self).__init__()
       self.conv1 = torch.nn.Sequential(
           torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
           torch.nn.BatchNorm2d(64),
           torch.nn.ReLU(inplace=True),
           torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
       )
       self.conv2_x = copy_res(64, 64, S=1, cnt=2)
       self.conv3_x = copy_res(64, 128, S=2, cnt=2)
       self.conv4_x = copy_res(128, 256, S=2, cnt=2)
       self.conv5_x = copy_res(256, 512, S=2, cnt=2)

       self.average_pool = torch.nn.AvgPool2d(7, stride=1)
       self.FC = torch.nn.Linear(512, 101)
       self.dropout = torch.nn.Dropout(0.3)
   
   # 前向计算（输入为x）
   def forward(self, x):
       conv1_out = self.conv1(x)
       conv2_out = self.conv2_x(conv1_out)
       conv3_out = self.conv3_x(conv2_out)
       conv4_out = self.conv4_x(conv3_out)
       conv5_out = self.conv5_x(conv4_out)
       res = self.average_pool(conv5_out)
       res = res.view(res.size(0), -1)
       res = self.FC(res)
       return self.dropout(res)

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
       model = Net().cuda()
   else:
       model = Net()
   #initNetParams(model)
   print(model)
   optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4) 
   loss_func = torch.nn.CrossEntropyLoss()

   for epoch in range(200):  # 训练200批次
       print('epoch {}'.format(epoch + 1))
       for p in optimizer.param_groups:
          p['lr'] *= 0.98
       # 训练
       model.train()
       train_loss = 0.
       train_acc = 0.
       cnt = 0
       total = 0
       for batch_x, batch_y in train_loader:  # 特征 标号
           if is_cuda:
               batch_x, batch_y = Variable(batch_x).cuda(), Variable(batch_y).cuda()
           else:
               batch_x, batch_y = Variable(batch_x), Variable(batch_y)
           #print(batch_x)
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
       total += len(batch_x)
   print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_data)), eval_acc / (len(test_data))))

if __name__ == '__main__':
    ssl._create_default_https_context = ssl._create_unverified_context
    resnet()
