import numpy as np
image = np.random.rand(4, 4)

# 3. Crop
def crop(image, start_row, start_col, height, width):
    return image[start_row:start_row+height, start_col:start_col+width]

cropped_image = crop(image, 1, 1, 2, 2)

print(image)
print(cropped_image)

"""실행결과
[[0.70174247 0.4574132  0.35357869 0.42901407]
 [0.06450882 0.22828803 0.56800398 0.07489028]
 [0.79159328 0.85046998 0.54123755 0.7476236 ]
 [0.56880354 0.7070836  0.42464435 0.12008319]]
[[0.22828803 0.56800398]
 [0.85046998 0.54123755]]
"""