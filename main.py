from ultralytics import YOLO
import cv2
import easyocr
import sys
import os


MODEL_YOLU = r"C:\Users\necmi\PycharmProjects\PlakaTanima\runs\detect\train\weights\best.pt"
DEFAULT_RESIM = r"C:\Users\necmi\PycharmProjects\PlakaTanima\deneme.jpg"
DEFAULT_VIDEO = r"C:\Users\necmi\PycharmProjects\PlakaTanima\deneme.webm"


# Ekran Boyutlandırma Fonksiyonu
def goruntu_boyutlandir(img, hedef_genislik=1024):


    h, w = img.shape[:2]
    if w > hedef_genislik:
        oran = hedef_genislik / w
        yeni_yukseklik = int(h * oran)
        img = cv2.resize(img, (hedef_genislik, yeni_yukseklik))
    return img


def islem_resim(model, reader):
    print("\n--- RESİM MODU ---")
    img_path = input(f"Resim yolunu girin (Varsayılan için Enter'a bas: {DEFAULT_RESIM}): ")
    if img_path.strip() == "":
        img_path = DEFAULT_RESIM

    if not os.path.exists(img_path):
        print(f"HATA: '{img_path}' dosyası bulunamadı! Ana menüye dönülüyor...")
        return

    img = cv2.imread(img_path)
    print("Resim işleniyor...")

    # Tahmin Yap
    results = model(img)

    for result in results:
        boxes = result.boxes.data.tolist()
        for box in boxes:
            x1, y1, x2, y2, score, class_id = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Plakayı kes ve oku
            plaka_crop = img[y1:y2, x1:x2]
            gray_crop = cv2.cvtColor(plaka_crop, cv2.COLOR_BGR2GRAY)
            ocr_result = reader.readtext(gray_crop, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')

            if len(ocr_result) > 0:
                text = " ".join([res[1] for res in ocr_result])
                text = text.replace('TRI', '').replace('TR', '').strip()

                print(f"-> Tespit Edilen Plaka: {text}")

                # Çizim
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Resmi ekrana sığacak şekilde küçült
    img_resized = goruntu_boyutlandir(img)

    cv2.imshow("RESIM SONUCU (Kapatmak icin herhangi bir tusa basin)", img_resized)
    cv2.waitKey(0)  # Tuşa basılmasını bekle
    cv2.destroyAllWindows()
    print("Resim kapatıldı. Ana menüye dönülüyor...")


def islem_video(model, reader):
    print("\n--- VIDEO MODU ---")
    vid_path = input(f"Video yolunu girin (Varsayılan için Enter'a bas: {DEFAULT_VIDEO}): ")
    if vid_path.strip() == "":
        vid_path = DEFAULT_VIDEO

    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print(f"HATA: '{vid_path}' dosyası açılamadı! Ana menüye dönülüyor...")
        return

    print("Video başlatılıyor... Çıkmak için 'q' tuşuna basın.")

    track_history = {}
    frame_count = 0
    kare_atlatma = 5  # Hız optimizasyonu için

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video bitti.")
            break

        frame_count += 1

        # YOLO Tahmini
        results = model(frame, stream=True)

        for result in results:
            boxes = result.boxes.data.tolist()
            for box in boxes:
                x1, y1, x2, y2, score, class_id = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Koordinat güvenliği
                h_img, w_img, _ = frame.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img, x2), min(h_img, y2)

                # Çok küçükleri yoksay
                if (x2 - x1) < 50: continue

                plaka_id = f"{x1}_{y1}"
                current_text = track_history.get(plaka_id, "")

                # Performans için her 5 karede bir OCR yap
                if frame_count % kare_atlatma == 0:
                    plaka_crop = frame[y1:y2, x1:x2]
                    if plaka_crop.size > 0:
                        # İyileştirme (Upscale + Grayscale)
                        gray = cv2.cvtColor(plaka_crop, cv2.COLOR_BGR2GRAY)
                        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

                        ocr_result = reader.readtext(gray, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                        if len(ocr_result) > 0:
                            text = " ".join([res[1] for res in ocr_result])
                            text = text.replace('TRI', '').replace('TR', '').strip()
                            if len(text) > 3:
                                current_text = text
                                track_history[plaka_id] = text


                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if current_text:
                    cv2.putText(frame, current_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Videoyu ekrana sığacak şekilde boyutlandır
        frame_resized = goruntu_boyutlandir(frame, hedef_genislik=800)

        cv2.imshow("VIDEO TAKIP (Cikis icin 'q' basin)", frame_resized)

        if cv2.waitKey(250) & 0xFF == ord('q'):
            print("Kullanıcı çıkış yaptı.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Video kapatıldı. Ana menüye dönülüyor...")


def main():
    print("Sistem Başlatılıyor, Modeller Yükleniyor... (Lütfen Bekleyin)")
    # Modelleri döngünün dışında bir kez yüklüyoruz ki her seferinde bekletmesin
    try:
        model = YOLO(MODEL_YOLU)
        # GPU varsa True, yoksa False
        reader = easyocr.Reader(['en'], gpu=True)
        print(">> Modeller Başarıyla Yüklendi.")
    except Exception as e:
        print(f"HATA: Modeller yüklenemedi. Yol yanlış olabilir.\nHata detayı: {e}")
        return

    while True:
        print("\n" + "=" * 40)
        print("   ARAÇ PLAKA TANIMA SİSTEMİ (ANA MENÜ)   ")
        print("=" * 40)
        print(" [1] Resim Üzerinde Test Et")
        print(" [2] Video Üzerinde Test Et")
        print(" [3] Programdan Çıkış (Kapat)")
        print("=" * 40)

        secim = input("Seçiminiz (1-3): ")

        if secim == '1':
            islem_resim(model, reader)
        elif secim == '2':
            islem_video(model, reader)
        elif secim == '3' or secim.lower() == 'q':
            print("Program sonlandırılıyor. İyi günler!")
            sys.exit()
        else:
            print("Geçersiz seçim! Lütfen 1, 2 veya 3'e basın.")


if __name__ == "__main__":
    main()