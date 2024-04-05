import numpy as np
image = np.random.rand(4, 4) # 가상의 이미지 데이터 생성 (4x4 픽셀, 흑백)

# 1. Max Pooling : 차원 축소 / 가장 큰 응답값만 사용
def max_pooling(image, pool_size):
    output_size = np.array(image.shape) // pool_size
    pooled_image = np.zeros(output_size)
    for i in range(output_size[0]):
        for j in range(output_size[1]):
            pooled_image[i, j] = np.max(image[i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size])
    return pooled_image

pooled_image = max_pooling(image, 2)

print(image)
print(pooled_image)

"""실행결과
[[0.45762618 0.56599022 0.78862062 0.04887893]
 [0.8933713  0.15056131 0.1261243  0.24107171]
 [0.23022256 0.98836642 0.77957824 0.54870795]
 [0.19010102 0.57122089 0.19503498 0.37152891]]
[[0.8933713  0.78862062]
 [0.98836642 0.77957824]]
"""