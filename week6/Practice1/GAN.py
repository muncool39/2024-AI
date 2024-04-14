import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import matplotlib.pyplot as plt

# 데이터셋 로딩
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(14),
    transforms.Normalize((0.5,), (0.5,)),
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

# 생성자(G) 정의
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 밑에서 정의한 크기(100)의 임의이 Noise로부터 이미지 생성
            nn.Linear(100, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 196),
            # 활성화 함수 Tanh (-1~1) 사용
            nn.Tanh()
        )
    def forward(self, input):
        # view 를 거쳐 반환
        return self.main(input).view(-1, 1, 14, 14)

# 감별자(D) 정의
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 1차원 벡터 -> 256 변환
            nn.Linear(14*14, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            # 256 -> 1 변환
            nn.Linear(256, 1),
            # 활성화 함수 Sigmoid (0~1) 사용
            nn.Sigmoid()
        )

    def forward(self, input):
        input = input.view(-1, 14*14)
        return self.main(input)

# 모델 초기화
G = Generator().cuda()
D = Discriminator().cuda()

# 옵티마이저 설정
optimizerG = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 손실 함수 설정
criterion = nn.BCELoss()

# 학습 시작
epochs = 30
for epoch in range(epochs):
    for i, (images, _) in enumerate(train_loader):
        images = images.cuda()
        # 실제 데이터를 사용하여 판별자 학습
        D.zero_grad()
        # 실제 이미지를 넣었을 때 나오는 출력값이 1이 되게끔 할당
        real_labels = torch.ones(images.size(0), 1).cuda()
        real_output = D(images)
        d_loss_real = criterion(real_output, real_labels)

        # 가짜 데이터 생성하여 판별자 학습
        # 100 크기의 노이즈 생성
        noise = torch.randn(images.size(0), 100).cuda()
        # 노이즈를 생성자에 입력해 가짜 이미지 얻음
        fake_images = G(noise)
        fake_labels = torch.zeros(images.size(0), 1).cuda()
        # G 의 가중치를 없애기 위해 (G가 영향을 받지 않도록 하기 위한) detach
        fake_output = D(fake_images.detach())
        d_loss_fake = criterion(fake_output, fake_labels)

        # 판별자 손실 계산 및 업데이트
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizerD.step()

        # 생성자 학습
        G.zero_grad()
        # D 학습때 만든 가짜 이미지 사용
        output = D(fake_images)
        g_loss = criterion(output, real_labels)
        g_loss.backward()
        optimizerG.step()

    print(f'Epoch [{epoch+1}/{epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')

    with torch.no_grad():
        test_z = torch.randn(16, 100).cuda()
        generated_images = G(test_z)

        # 이미지 그리드 생성
        grid = torchvision.utils.make_grid(generated_images, nrow=4, normalize=True)
        plt.figure(figsize=(10,10))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')

        # 생성된 이미지 저장
        plt.savefig(f'sample_data/epoch_{epoch+1}.png')
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