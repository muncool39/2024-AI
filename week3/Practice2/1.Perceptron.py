import torch
import torch.nn as nn
import torch.optim as optim

# 퍼셉트론 모델 정의
class Perceptron(nn.Module):
  def __init__(self, input_size):
    super(Perceptron, self).__init__()
    # 출력 크기가 1인 Linear 레이블 생성
    self.linear = nn.Linear(input_size, 1) # 입력 크기에서 1개의 출력으로 매핑
  def forward(self, x): # 순전파
    return torch.sigmoid(self.linear(x)) # 활성화 함수로 시그모이드 사용

# AND 연산을 위한 학습 데이터 생성
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([[0.], [0.], [0.], [1.]])

# 모델 정의 (위에서 생성한 퍼셉트론)
model = Perceptron(2) # 입력 크기 2
# 손실 함수 정의
criterion = nn.BCELoss() # 이진 분류를 위한 Binary Cross-Entropy 손실
# 최적화 방법 정의, 학습률 설정
optimizer = optim.SGD(model.parameters(), lr=0.1) # 확률적 경사 하강법, lr = 학습률

# 모델 학습
epochs = 1000
for epoch in range(epochs):
  optimizer.zero_grad() # 그라디언트 초기화
  outputs = model(X) # 순전파 (모델 안에 forward)
  loss = criterion(outputs, y) # 손실 계산
  loss.backward() # 역전파
  optimizer.step() # 가중치 업데이트
  if (epoch+1) % 100 == 0:
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 학습된 모델 테스트
with torch.no_grad(): # 기울기 계산을 수행하지 않음
  y_predicted = model(X)
  y_predicted_cls = y_predicted.round() # 확률을 0 또는 1로 반올림
  accuracy = y_predicted_cls.eq(y).sum() / float(y.shape[0])
  print(f'Accuracy: {accuracy:.4f}')

"""출력결과
Epoch [100/1000], Loss: 0.4558
Epoch [200/1000], Loss: 0.3587
Epoch [300/1000], Loss: 0.2985
Epoch [400/1000], Loss: 0.2570
Epoch [500/1000], Loss: 0.2263
Epoch [600/1000], Loss: 0.2024
Epoch [700/1000], Loss: 0.1832
Epoch [800/1000], Loss: 0.1674
Epoch [900/1000], Loss: 0.1540
Epoch [1000/1000], Loss: 0.1426
Accuracy: 1.0000
"""