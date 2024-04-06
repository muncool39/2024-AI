from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt') # pretrained
# Run batched inference on a list of images
results = model(['./이미지 경로.jpg'])
# Process results list
for result in results:
  boxes = result.boxes # Boxes object
  masks = result.masks # Masks object
  keypoints = result.keypoints # Keypoints object
  probs = result.probs # Probs for classification
  result.show() # display to screen
  result.save(filename='./sample_data/result.jpg')