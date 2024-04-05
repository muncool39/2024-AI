import numpy as np
image = np.random.rand(4, 4)

# 2. Padding (Zero-padding)
def padding(image, pad_width):
    padded_image = np.zeros((image.shape[0] + 2 * pad_width, image.shape[1] + 2 * pad_width), dtype=image.dtype)
    padded_image[pad_width:-pad_width, pad_width:-pad_width] = image
    return padded_image

padded_image = padding(image, 1)

print(image)
print(padded_image)

"""실행결과
[[0.70174247 0.4574132  0.35357869 0.42901407]
 [0.06450882 0.22828803 0.56800398 0.07489028]
 [0.79159328 0.85046998 0.54123755 0.7476236 ]
 [0.56880354 0.7070836  0.42464435 0.12008319]]
[[0.         0.         0.         0.         0.         0.        ]
 [0.         0.70174247 0.4574132  0.35357869 0.42901407 0.        ]
 [0.         0.06450882 0.22828803 0.56800398 0.07489028 0.        ]
 [0.         0.79159328 0.85046998 0.54123755 0.7476236  0.        ]
 [0.         0.56880354 0.7070836  0.42464435 0.12008319 0.        ]
 [0.         0.         0.         0.         0.         0.        ]]
"""