import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 데이터 전처리를 위한 변환 정의
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((16, 16)),
     transforms.Normalize((0.5,), (0.5,))])

# 학습 데이터셋 로딩
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

# 테스트 데이터셋 로딩
testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

# 클래스 이름
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) # 16x16 -> 12x12
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(6, 16, 5) 
        # Resize 전 : self.fc1 = nn.Linear(16 * 4 * 4, 120) 
        self.fc1 = nn.Linear(16 * 1 * 1, 120) # Resize로 인한 크기 변동
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        # 16x16x1 -> 12x12x6 ->6x6x6
        x = self.pool(self.relu(self.conv1(x))) 
        # 6x6x6 -> 2x2x16 -> 1x1x16
        x = self.pool(self.relu(self.conv2(x)))
        # Resize 전 : x = x.view(-1, 16 * 4 * 4)
        x = x.view(-1, 16 * 1 * 1) # Resize로 인한 크기 변동
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # 데이터셋을 여러 번(10 epochs) 반복
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 2000 == 1999:    
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

correct = 0
total = 0
# 기울기를 계산하지 않아도 되므로, 메모리 소비를 줄이고 계산 속도를 높이기 위해 no_grad() 사용
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        # 가장 높은 값(energy)을 갖는 분류(class)를 선택합니다.
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total}%')

"""실행결과
[1, 2000] loss: 1.585
[1, 4000] loss: 0.752
[1, 6000] loss: 0.643
[1, 8000] loss: 0.602
[1, 10000] loss: 0.584
[1, 12000] loss: 0.537
[1, 14000] loss: 0.532
[2, 2000] loss: 0.489
[2, 4000] loss: 0.468
[2, 6000] loss: 0.485
[2, 8000] loss: 0.443
[2, 10000] loss: 0.440
[2, 12000] loss: 0.439
[2, 14000] loss: 0.439
Finished Training
Accuracy of the network on the 10000 test images: 83%
"""