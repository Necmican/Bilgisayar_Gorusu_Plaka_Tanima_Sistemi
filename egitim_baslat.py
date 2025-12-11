from ultralytics import YOLO

if __name__ == '__main__':
    # 1. Modeli Yükle (Transfer Learning)
    # yolov8n.pt en hızlısıdır. Daha güçlü olsun dersen 'yolov8s.pt' kullanabilirsin.
    model = YOLO('yolov8n.pt')

    # 2. Eğitimi Başlat
    # data: yaml dosyanın yolu
    # epochs: 50 tur dönecek (Sonuç iyi çıkmazsa 100 yaparsın)
    # imgsz: Resim boyutu (640 standarttır)
    # device: 0 ekran kartı (varsa), yoksa 'cpu'
    print("Eğitim başlıyor... Bu işlem bilgisayar hızına göre 1-2 saat sürebilir.")

    results = model.train(data=r'C:\Users\necmi\PycharmProjects\PlakaTanima\yolo_dataset\data.yaml')