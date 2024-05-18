from ultralytics import YOLO

model = YOLO('yolov8s.pt')
results = model.train(data='dataset/data.yaml', epochs=100, batch=8, imgsz=640, project='runs/train', name='Sino-nom')
