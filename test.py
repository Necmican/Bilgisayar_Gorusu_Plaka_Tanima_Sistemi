from ultralytics import YOLO
import cv2
import easyocr

# --- 1. AYARLAR VE YOLLAR ---
# Senin bilgisayarındaki Yollar (Path)
model_yolu = r"C:\Users\necmi\PycharmProjects\PlakaTanima\runs\detect\train\weights\best.pt"
resim_yolu = r"C:\Users\necmi\PycharmProjects\PlakaTanima\deneme.jpg"

print("Sistem hazırlanıyor (Model ve OCR yükleniyor)...")

# --- 2. MODELLERİN YÜKLENMESİ ---
# YOLO Modelini Yükle
model = YOLO(model_yolu)

# EasyOCR'ı Yükle (Sadece İngilizce yeterli, gpu=False işlemciyi kullanır)
reader = easyocr.Reader(['en'], gpu=False)

# --- 3. RESİM İŞLEME ---
# Resmi oku
img = cv2.imread(resim_yolu)

# YOLO ile tahmin yap
results = model(img)

# Sonuçları işle
for result in results:
    boxes = result.boxes.data.tolist()  # Koordinatları al

    for box in boxes:
        # Koordinatları ayrıştır
        x1, y1, x2, y2, score, class_id = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # --- A. KESME (CROP) ---
        # Resimden sadece plaka olan dikdörtgeni kesip alıyoruz
        plaka_crop = img[y1:y2, x1:x2]

        # --- B. İYİLEŞTİRME ---
        # Kesilen plakayı Siyah-Beyaz yap (OCR başarısını artırır)
        gray_crop = cv2.cvtColor(plaka_crop, cv2.COLOR_BGR2GRAY)

        # --- C. OKUMA (OCR) ---
        # allowlist: Sadece harf ve rakamları okumasına izin veriyoruz.
        ocr_result = reader.readtext(gray_crop, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')

        # --- D. METNİ BİRLEŞTİRME VE TEMİZLEME ---
        if len(ocr_result) > 0:
            # EasyOCR bazen "66", "AP" diye ayrı ayrı bulur, bunları birleştiriyoruz:
            text = " ".join([res[1] for res in ocr_result])

            # 'TR', 'TRI' gibi kısımları temizle (İsteğe bağlı)
            text = text.replace('TRI', '').replace('TR', '').strip()

            print(f"-> Okunan Plaka: {text}")

            # Ekrana Çizdirme
            # Yeşil kutu çiz
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Üstüne metni yaz
            cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)
        else:
            print("Plaka konumu bulundu ama yazı okunamadı.")

# --- 4. SONUCU EKRANDA GÖSTERME ---
# Resim çok büyükse ekrana sığması için küçültüyoruz
h, w = img.shape[:2]
new_width = 800
new_height = int(h * (new_width / w))
resized_img = cv2.resize(img, (new_width, new_height))

cv2.imshow("Plaka Tespit Sonucu", resized_img)

# Bir tuşa basana kadar bekle
cv2.waitKey(0)
cv2.destroyAllWindows()