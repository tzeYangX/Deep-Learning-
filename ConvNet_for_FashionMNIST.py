# PyTorch MNIST Example 参见https://github.com/pytorch/examples
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pylab

base_dir='/home/kesci/input/fashion_mnist5774/fashion-mnist/'
train_set = datasets.FashionMNIST(  # 
    base_dir+'fashion',  # 数据集位置
    train=True,  # 训练集
    download=False,  # 下载
    transform=transforms.Compose(  # 变换
        [
            transforms.ToTensor(),  # 从[0,255]归一化到[0,1]
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]))  # 进一步按照均值方差归一化
test_set = datasets.FashionMNIST(
    base_dir+'fashion',
    train=False,  # 测试集
    transform=transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))]))
train_loader = torch.utils.data.DataLoader(  # 提取器
    train_set, batch_size=100, shuffle=True)  # batch_size 批处理大小 shuffle 是否打乱顺序

test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=1000, shuffle=False)

pylab.rcParams['figure.figsize'] = (4.0, 4.0) # plot size
print(train_set)
print(test_set)
img, lbl = iter(test_loader).next()  # 深度学习框架一般按照[n c h w]的顺序排列数据
print(img.shape)
img = img[0, :, :].data.numpy().squeeze(0)  # 转化为numpy格式并去除额外维度
plt.imshow(img, cmap='jet')
plt.show()



class LeNet5(nn.Module):  # 参考lenet5网络结构 形似
    def __init__(self):
        super(LeNet5, self).__init__()
        self.C1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=5), nn.ReLU())  # 卷积层 + ReLU 激活函数
        self.S2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 最大值池化
        self.C3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3), nn.ReLU())
        self.S4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.C5 = nn.Sequential(nn.Linear(1600, 256), nn.ReLU())  # 全连接层
        self.F6 = nn.Sequential(nn.Linear(256, 128), nn.ReLU())
        self.OUT = nn.Linear(128, 10)

    def forward(self, x):
        x = self.C1(x)
        x = self.S2(x)
        x = self.C3(x)
        x = self.S4(x)
        x = x.view(-1, 1600)  # 将图展开成向量
        x = self.C5(x)
        x = self.F6(x)
        x = self.OUT(x)
        return F.log_softmax(x, dim=1)  # softmax输出 用于分类任务
print(LeNet5())



torch.manual_seed(2018)  # 设置随机数种子
model = LeNet5()  # 实例化模型
#optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9) # 学习率、动量修改
optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
#optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.97, eps=1e-08, weight_decay=0, momentum=0, centered=False)
#optimizer=optim.Adadelta(model.parameters(), lr=1.0, rho=0.95, eps=1e-06, weight_decay=0)
n_epoch = 30  # 修改迭代更多次
for epoch in range(1, n_epoch + 1):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader): #枚举函数，枚举数据集中的数据
        optimizer.zero_grad()  # 梯度清空
        output = model(data)  # 前向推理
        #print(output)
        #print(batch_idx)
        loss = F.cross_entropy(output, target) # 交叉熵损失函数
        #loss = F.nll_loss(output, target)  # 非负极大似然损失函数
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 参数调整
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    model.eval()  # 测试模式
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.cross_entropy(
                output, target, size_average=False).item()  # 计算每一组的loss
            #test_loss += F.nll_loss(
                #output, target, size_average=False).item()  # 计算每一组的loss
            pred = output.max(
                1, keepdim=True)[1]  # 获取索引
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
                     
  
# 创建提取特征类
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name is "C5":
                x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs
print(FeatureExtractor(model,['C1','C2']))


# 提取数据
test_loader_for_feature = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)
img, label = iter(test_loader_for_feature).next()
exactor = FeatureExtractor(model, ['C1', 'C3'])
features = exactor(img)


# 可视化
pylab.rcParams['figure.figsize'] = (10.0, 10.0) # plot size
for i in range(6):
    ax = plt.subplot(2, 3, i + 1)
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    plt.imshow(features[0].data.numpy()[0, i, :, :], cmap='jet')
plt.show()
for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    plt.imshow(features[1].data.numpy()[0, i, :, :], cmap='jet')
plt.show()
