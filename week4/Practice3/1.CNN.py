import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 이미지가 들어오면 어느 카테고리(클래스)에 해당되는지 맞추는 모델
# 내장 데이터를 사용할 경우의 코드 (커스텀 데이터일 경우는 코드 다름)

# 데이터 전처리를 위한 변환 정의
transform = transforms.Compose(
    [transforms.ToTensor(),
     #transforms.Resize((16, 16)),
     transforms.Normalize((0.5,), (0.5,))])

# 학습 데이터셋 로딩
# torchvision 에 내장된 데이터 불러옴
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                        download=True, transform=transform)
# batch_size : 한번에 몇개씩 데이터를 입력할 건지
# shuffle : 데이터를 무작위로 섞을 것인지 (학습할 땐 섞는 것이 좋다)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

# 테스트 데이터셋 로딩
testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                       download=True, transform=transform)
# 테스트 할 땐 shuffle 하지 않는다
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

# 클래스 이름
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# CNN 모델
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 컨볼루션 필터 설정
        # nn.Conv2d(입력 채널 수, 출력 채널 수, 커널 크기)
        # 입력 채널 수 : 이미지 컬러 정보 (흑백 1, 컬러 3)
        # 이미지 변환 공식
        # ( 이미지 크기 - 커널 크기 + 2*패딩 ) / 스트라이드 + 1
        self.conv1 = nn.Conv2d(1, 6, 5) # 28x28 -> 24x24 (28-5+1)
        # 차원 축소
        self.pool = nn.MaxPool2d(2, 2) # 24x24 -> 12x12, 8x8 -> 4x4 
        self.relu = nn.ReLU()

        # 위 conv1 의 출력 채널 수 = conv2 의 입력 채널 수
        # 레이어 연결되는 부분 - 전 레이어의 출력 크기와 입력 크기를 맞추어 줘야 함
        self.conv2 = nn.Conv2d(6, 16, 5) # 12x12 -> 8x8
        self.fc1 = nn.Linear(16 * 4 * 4, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) # 10개 클래스 분류

    def forward(self, x):
        # 1.컨볼루션 필더 통과 2.ReLu 통과 3.차원 축소를 위한 PollingLayer 통과
        # 처음 입력 : 28x28x1 -> 컨볼루션 필터 : 24x24x6 -> polling : 12x12x6 
        x = self.pool(self.relu(self.conv1(x))) 
        # 과정 한번 더 반복
        # 12x12x6 -> 8x8x16 -> 4x4x16 
        x = self.pool(self.relu(self.conv2(x)))

        # 이미지 형태를 펼쳐주는 Flatten(평탄화) 작업 진행
        # 위에서 받은 변화된 크기 4x4x16 -> 256
        x = x.view(-1, 16 * 4 * 4)
        # MLP 작업 진행
        x = self.relu(self.fc1(x)) # 256 -> 120
        x = self.relu(self.fc2(x)) # 120 -> 84
        x = self.fc3(x) # 84 -> 10
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 학습
for epoch in range(2):  # 데이터셋 반복
    running_loss = 0.0
    # 한번에 통과시킬 수 없어서 끊어서 진행 (batch_size)
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

# loss 가 점점 줄어들어야 한다
"""출력결과
[1, 2000] loss: 1.464
[1, 4000] loss: 0.681
[1, 6000] loss: 0.570
[1, 8000] loss: 0.518
[1, 10000] loss: 0.475
[1, 12000] loss: 0.451
[1, 14000] loss: 0.438
[2, 2000] loss: 0.402
[2, 4000] loss: 0.388
[2, 6000] loss: 0.386
[2, 8000] loss: 0.351
[2, 10000] loss: 0.381
[2, 12000] loss: 0.359
[2, 14000] loss: 0.354
Finished Training
Accuracy of the network on the 10000 test images: 86%
"""