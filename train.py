from ultralytics import YOLO

if __name__ == '__main__':
    # model.load('yolo11n.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    #/root/ultralytics/ultralytics/cfg/models/v8/yolov8-new-EMBSFPN.yaml
    model = YOLO(model=r'/root/ultralytics/runs/train/yolov8-imp24/weights/last.pt')
    model.train(data=r'data.yaml',
                imgsz=640,
                epochs=300,
                batch=64,
                workers=8,
                device='',
                optimizer='SGD',
                close_mosaic=0,
                resume=True,
                project='runs/train',
                name='yolov8-imp2',
                single_cls=False,
                cache=False,
                )
