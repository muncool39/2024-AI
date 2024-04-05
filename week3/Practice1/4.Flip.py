import numpy as np
image = np.random.rand(4, 4)

def flip_image(image, axis):
  # Assuming image is a 2D or 3D numpy array
  if axis == 'horizontal':
    # Flip horizontally
    return image[:, ::-1]
  elif axis == 'vertical':
    # Flip vertically
    return image[::-1,:]
  else:
    raise ValueError("Axis must be 'horizontal' or 'vertical'")

flipped_image = flip_image(image, 'horizontal')
print(image)
print(flipped_image)

"""실행결과
[[0.57067209 0.09886472 0.13716988 0.44084035]
 [0.57921204 0.64162615 0.17742624 0.12162747]
 [0.96254964 0.25158789 0.5718381  0.86937841]
 [0.20856677 0.36389274 0.95050391 0.4861801 ]]
[[0.44084035 0.13716988 0.09886472 0.57067209]
 [0.12162747 0.17742624 0.64162615 0.57921204]
 [0.86937841 0.5718381  0.25158789 0.96254964]
 [0.4861801  0.95050391 0.36389274 0.20856677]]
"""