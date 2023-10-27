import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        # convolution layer1
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5,stride=1)
        # pool layer1
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        # convolution layer2
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1)
        # pool layer2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # fully connected layer1
        self.fc1 = nn.Linear(32*5*5,120)
        # fully connected layer2
        self.fc2 = nn.Linear(120,84)
        # fully connected layer3
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        # convolution -> activation function -> pool
        x = F.relu(self.conv1(x))  # conv(3*32*32) -> activation(16*28*28)
        x = self.pool1(x)  # activation -> pool
        x = F.relu(self.conv2(x))
        x = self.pool2(x) 
        # reshape the output size automatically into 32*5*5
        x = x.view(-1, 32 * 5 * 5)  
        # fully connected -> activation function
        x = F.relu(self.fc1(x))  # output(120)
        x = F.relu(self.fc2(x))  # output(84)
        x = self.fc3(x)  # output(10)
        return x

# 从`torchvision.datasets`中下载数据集，包括训练集和测试集。
# 
# 使用`transforms.Compose`拼接函数，使用`ToTensor`将Image转化为tensor格式，然后使用`normalize`将tensor标准化/归一化，加快模型收敛。

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# load data
trainloader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)
testloader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False)
# the dataset's classes names
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# batch大小设置为4张图片。
# 
# 查看数据集中的数据情况，打印出一个batch。从数据集中直接获取到的image是tensor形式，需要反标准化将tensor转换成图像。使用`show`函数实现。

def show(img):
    img = img / 2 + 0.5 
    npimg = img.numpy()
    # pytorch presentation: [channel, height, width]
    # plt use numpy: [height, width, channel]
    # use np.transpose() to change axis order
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(trainloader)
images, labels = dataiter.__next__()  # return 4 pics and labels
print(' '.join('%11s' % classes[labels[j]] for j in range(4)))  # print labels
show(torchvision.utils.make_grid((images + 1) / 2))


# 使用交叉熵损失函数，同时选择Adam优化器优化，学习率设为0.001
# use cuda if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = LeNet()
net_model = net.to(device)

loss_function = nn.CrossEntropyLoss()
# choose Adam to optimize, learning rate=0.001
optimizer = optim.Adam(net.parameters(), lr=0.001)

max_epoch = 100

# 训练总共进行100轮，训练过程中计算每一轮训练的平均损失和总体准确率，绘制出损失函数变化和整体准确率变化。

loss_per_epoch = []
accuracy_per_epoch = []

for epoch in range(max_epoch):
    print("Begin epoch %i"% (epoch+1))
    Loss = []
    Accuracy = []
    total = 0
    correct = 0
    for i, data in enumerate(trainloader,0):
        inputs, labels = data
        # load input and label data into gpu
        inputs, labels = inputs.to(device), labels.to(device)
        # make gradient zero
        optimizer.zero_grad()
        # forward
        output = net_model(inputs)
        loss = loss_function(output,labels)

        # get current predict to show training status
        predict = output.argmax(dim=1)
        total += labels.size(0)
        correct += (predict==labels).sum().item()

        # backward, generate gradient automatically
        loss.backward()
        optimizer.step()

        if i % 2000 == 1999:
            Loss.append(loss.item())

    loss_per_epoch.append(np.array(Loss).mean())
    accuracy_per_epoch.append(correct/total)
    print("Epoch %(epoch)i  loss: %(loss)f, accuracy: %(accr)f" \
          % {"epoch":epoch+1,"loss":np.array(Loss).mean(),"accr":correct/total})
print("Train finished.")

# 将模型训练过程loss和accuracy的变化用matplotlib绘制出来。

epochs = [i+1 for i in range(max_epoch)]

plt.subplot(211)
plt.plot(epochs,loss_per_epoch)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Loss Per Epoch")
plt.show()

plt.subplot(212)
plt.plot(epochs,accuracy_per_epoch)
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.title("Total Accuracy Per Epoch")
plt.show()

# 平均loss的波动较大，整体趋势随着训练轮数增加减小。训练集上的准确率随着训练增加，但是在达到80%左右后不断波动，没有进一步提升。可能存在过拟合的风险。
# 
# 打印出模型基本信息并将模型权重文件保存到文件目录中。

print(net_model)
torch.save(net_model.state_dict(), './models/model-cifar10-100.pth')

# 计算模型在10000张测试集上的准确率。
correct = 0
total = 0
# torch.no_grad() do not calculate gradient
# do not backward
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net_model(images)
        # torch.max(outputs.data, 1)returns the max index
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
 
print('Accuracy of the network: %d %%' % (100 * correct / total))
 
# print accuracy for each class
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10)) 
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net_model(images) # outputs dimension：4*10
        _, predicted = torch.max(outputs, 1) # predicted dimension: 4*1
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
for i in range(10):
    print('Accuracy of %5s : %f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

# 网络整体在测试集上的正确率为65%，每个分类的正确率基本上在60%-70%左右。
# 
# 选择一张不在数据集中的图片进行测试。

# load the trained model
model = LeNet()
model.load_state_dict(torch.load('./models/model-cifar10.pth'))

pic = Image.open('test_cat.jpg')
plt.imshow(pic)
plt.axis('off') 
plt.show() # show original picture
    
# use trans to format input pic
trans = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
img = trans(pic)
# expand the input into 4 dimension: [batchsize, channel, width, height]
img = img.unsqueeze(0)

with torch.no_grad():
    output = model(img)
    predict = torch.max(output, dim=1)[1]
print(classes[predict.item()])


