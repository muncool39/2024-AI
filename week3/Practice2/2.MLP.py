import torch
import torch.nn as nn
import torch.optim as optim

# MLP (Multi-Layer Perceptron) 모델 정의
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(input_size, 4) # 입력층에서 은닉층으로의 매핑 (입력 크기 2, 은닉층 크기 2)
        self.activation = nn.Sigmoid() # 활성화 함수로 시그모이드 사용
        self.output = nn.Linear(4, 1) # 은닉층에서 출력층으로의 매핑 (은닉층 크기 2, 출력 크기 1)

    def forward(self, x):
        # 은닉층에 활성화 함수 적용
        x = self.hidden(x)
        x = self.activation(x)
        # 출력층에 활성화 함수 적용
        x = self.output(x)
        x = self.activation(x) 
        return x

X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([[0.], [0.], [0.], [1.]])


model = MLP(2) # 모델 변경
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 모델 학습
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()  # 그라디언트 초기화
    outputs = model(X)  # 순전파
    loss = criterion(outputs, y)  # 손실 계산
    loss.backward()  # 역전파
    optimizer.step()  # 가중치 업데이트
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 학습된 모델 테스트
with torch.no_grad():  # 기울기 계산을 수행하지 않음
    y_predicted = model(X)
    y_predicted_cls = y_predicted.round()  # 확률을 0 또는 1로 반올림
    accuracy = y_predicted_cls.eq(y).sum() / float(y.shape[0])
    print(f'Accuracy: {accuracy:.4f}')

"""실행결과
Epoch [100/1000], Loss: 0.5340
Epoch [200/1000], Loss: 0.5044
Epoch [300/1000], Loss: 0.4632
Epoch [400/1000], Loss: 0.4143
Epoch [500/1000], Loss: 0.3638
Epoch [600/1000], Loss: 0.3153
Epoch [700/1000], Loss: 0.2703
Epoch [800/1000], Loss: 0.2297
Epoch [900/1000], Loss: 0.1941
Epoch [1000/1000], Loss: 0.1637
Accuracy: 1.0000
"""