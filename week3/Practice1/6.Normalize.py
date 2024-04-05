import numpy as np
image = np.random.rand(4, 4)

# 6. Normalization : 정규화
def normalize(image):
  return (image - np.min(image)) / (np.max(image) - np.min(image))

normalized_image = normalize(image)

print(image)
print(normalized_image)

"""실행결과
[[0.96489894 0.81682255 0.14329642 0.16802022]
 [0.32198478 0.44528505 0.82821729 0.80487912]
 [0.5635156  0.25303997 0.31412421 0.67783781]
 [0.86342642 0.83423511 0.84838066 0.97298864]]
[[0.99024976 0.81177828 0.         0.02979876]
 [0.21536704 0.36397669 0.82551199 0.79738328]
 [0.50647598 0.13227019 0.20589296 0.64426467]
 [0.86794836 0.83276505 0.84981421 1.        ]]
"""