import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import matplotlib.pyplot as plt

# 데이터셋 로딩 (이미지 크기 28x28)
transform = transforms.Compose([
    # transforms.Resize(14),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

# 생성자 정의
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # 잠재 벡터를 7x7x256 크기의 특징 맵으로 변환
            # 처음 계산은 1x1 부터
            nn.ConvTranspose2d(100, 256, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 28x28 사이즈로 변경됨에 따른 변경 (업샘플링 한번 더 함)
            # 14x14 사이즈의 경우 여기서 바로 1채널로 끝냄
            # nn.ConvTranspose2d(256, 1, kernel_size=4, stride=2, padding=1)

            # 7x7x256에서 14x14x128 크기로 업샘플링
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 14x14x128에서 최종적으로 28x28x1 이미지를 생성 (1채널로 끝내기)
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1),

            nn.Tanh()
        )
    def forward(self, z):
        # input 크기 (1x1)
        z = z.view(-1, 100, 1, 1)
        img = self.model(z)
        return img


# 감별자 정의
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            # 크기 변동에 따른 연산 변경
            # nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            # nn.Conv2d(64, 1, kernel_size=3, stride=2, padding=0),
            nn.Conv2d(64, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid()
        )

    def forward(self, img):
        validity = self.model(img)
        return validity.view(-1, 1)


# 모델, 옵티마이저 및 손실 함수 초기화
G = Generator().cuda()
D = Discriminator().cuda()
optimizerG = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# 학습 시작
epochs = 30
for epoch in range(epochs):
    for i, (imgs, _) in enumerate(train_loader):
        imgs = imgs.cuda()

        # 실제 이미지와 가짜 이미지에 대한 레이블 준비
        real_labels = torch.ones(imgs.size(0), 1).cuda()
        fake_labels = torch.zeros(imgs.size(0), 1).cuda()

        # 실제 이미지를 사용하여 판별자 학습
        optimizerD.zero_grad()
        real_loss = criterion(D(imgs), real_labels)
        real_loss.backward()

        # 가짜 이미지 생성하여 판별자 학습
        z = torch.randn(imgs.size(0), 100).cuda()
        fake_imgs = G(z)
        fake_loss = criterion(D(fake_imgs.detach()), fake_labels)
        fake_loss.backward()
        optimizerD.step()

        # 생성자 학습
        optimizerG.zero_grad()
        g_loss = criterion(D(fake_imgs), real_labels)
        g_loss.backward()
        optimizerG.step()

    print(f'Epoch [{epoch+1}/{epochs}], d_loss: {real_loss.item() + fake_loss.item():.4f}, g_loss: {g_loss.item():.4f}')

    with torch.no_grad():
        test_z = torch.randn(16, 100).cuda()
        generated_images = G(test_z)

        # 이미지 그리드 생성
        grid = torchvision.utils.make_grid(generated_images, nrow=4, normalize=True)
        plt.figure(figsize=(10,10))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')

        # 생성된 이미지 저장
        plt.savefig(f'conv_epoch_{epoch+1}.png')
        plt.close()

# 학습된 모델로부터 이미지 생성
with torch.no_grad():
    test_noise = torch.randn(16, 100).cuda()
    generated_images = G(test_noise)
    generated_images = generated_images.cpu()
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(generated_images[i].reshape(14, 14), cmap='gray')
        plt.axis('off')
    plt.show()