from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO(model=r'/root/ultralytics/yolo11n.pt')  
    model.predict(source=r'/root/yolo/dataset/test/0001007.png',
                  save=True,
                  show=True,
                  )
