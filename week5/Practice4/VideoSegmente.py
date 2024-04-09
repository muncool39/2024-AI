import cv2
import torch
import numpy as np
from torchvision import models
from torchvision.transforms import functional as F

# 사전 훈련된 세그멘테이션 모델 로드
model = models.segmentation.deeplabv3_resnet50(pretrained=True)
model.eval()

# CUDA 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def segment_people(frame):
    # 이미지를 모델 입력으로 변환
    input_tensor = F.to_tensor(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    output_predictions = output.argmax(0)

    # '사람(person)' 클래스는 COCO 데이터셋에서 클래스 인덱스 15에 해당
    mask = (output_predictions == 15).cpu().numpy()
    
    # 마스크를 이용해 사람만 추출
    result = frame * mask[:, :, np.newaxis]
    return result

# 비디오 파일 로드
video_path = 'vid/video1.mp4'
cap = cv2.VideoCapture(video_path)

# 결과 비디오 설정
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('people_segmented.avi', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

print(cap.isOpened())

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # BGR에서 RGB로 변환
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 세그멘테이션 적용
    segmented_frame = segment_people(frame_rgb)
    
    # RGB에서 BGR로 다시 변환하여 저장/표시
    segmented_frame_bgr = cv2.cvtColor(segmented_frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
    out.write(segmented_frame_bgr)

    # 결과 프레임 보여주기 (로컬 PC에서 작동시)
    # cv2.imshow('frame', segmented_frame_bgr)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()
