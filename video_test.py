from ultralytics import YOLO
import cv2
import easyocr
import time

# --- 1. AYARLAR ---
model_yolu = r"C:\Users\necmi\PycharmProjects\PlakaTanima\runs\detect\train\weights\best.pt"
video_yolu = r"C:\Users\necmi\PycharmProjects\PlakaTanima\deneme.webm"  # BURAYA VIDEO YOLUNU YAZ

print("Model ve OCR yükleniyor...")
model = YOLO(model_yolu)
reader = easyocr.Reader(['en'], gpu=False)  # Eğer NVIDIA kartın varsa gpu=True yap

# Videoyu başlat
cap = cv2.VideoCapture(video_yolu)

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Video bitti

    # --- 2. TAHMİN VE İŞLEME ---
    results = model(frame, stream=True)  # stream=True videoda daha hızlıdır

    for result in results:
        boxes = result.boxes.data.tolist()

        for box in boxes:
            x1, y1, x2, y2, score, class_id = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Plakayı kes
            # Hata almamak için koordinatların resim sınırları içinde olduğundan emin oluyoruz
            h, w, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            plaka_crop = frame[y1:y2, x1:x2]

            # Kesilen parça boş değilse işlem yap
            if plaka_crop.size > 0:
                # Siyah-beyaz yap
                gray_crop = cv2.cvtColor(plaka_crop, cv2.COLOR_BGR2GRAY)

                # Oku
                ocr_result = reader.readtext(gray_crop, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')

                if len(ocr_result) > 0:
                    text = " ".join([res[1] for res in ocr_result])
                    text = text.replace('TRI', '').replace('TR', '').strip()

                    # Ekrana yaz
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 2)

    # --- 3. GÖSTERME ---
    # Videoyu ekrana sığacak boyuta getir (Opsiyonel)
    frame_resized = cv2.resize(frame, (1024, 768))

    cv2.imshow("Plaka Takip", frame_resized)

    # 'q' tuşuna basınca çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()