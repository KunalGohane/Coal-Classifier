from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')  # loading of a pretrained model 

model.train(data='E:/coal_dataset',
            epochs=100, imgsz=64)