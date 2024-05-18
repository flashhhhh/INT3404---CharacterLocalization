from ultralytics import YOLO

model = YOLO('best2.pt')
results = model.train(data='dataset/data.yaml', epochs=5, batch=8, imgsz=640, project='runs/train', name='Sino-nom')