# Lec6 인공지능 응용 실습 (3) 생성
- GAN
- DCGAN
- 모델 학습 과정
  - 필요 데이터: 이미지, 진짜레이블(1), 가짜레이블(0), 가짜이미지(G로 생성)
  - D: 가짜이미지 입력시 0이 예측되도록, 진짜이미지 입력시 1이 예측되도록 학습
  - G: 가짜이미지를 D로 판별했을 때, 1이 예측되도록 학습

## 실습 내용
### 실습 1
MNIST 데이터를 생성하는 GAN 학습

### 실습 2
MNIST 데이터를 생성하는 DCGAN 학습