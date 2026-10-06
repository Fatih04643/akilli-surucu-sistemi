# Akıllı Sürücü Güvenlik ve Dikkat Takip Sistemi (ADAS)

Bu proje, monoküler kamera akışı üzerinden sürücü yorgunluğunu, mikro uykuyu, dikkat dağınıklığını ve kural dışı akıllı telefon kullanımını gerçek zamanlı tespit eden derin öğrenme ve bilgisayarlı görü tabanlı bir erken uyarı sistemidir.

## Temel Özellikler
- **Yüz ve Göz Analizi:** MediaPipe Face Mesh tabanlı 468 landmark ve iris takibi.
- **Geometrik Metrikler:** EAR (Göz Açıklık Oranı), MAR (Ağız Açıklık Oranı) ve solvePnP ile Baş Pozu (Pitch, Yaw, Roll).
- **Kural İhlali Tespiti:** YOLOv8n ile gerçek zamanlı telefon kullanımı yakalama.
- **Zaman Serisi Karar Motoru:** Yanlış alarmları önleyen kayan pencere (sliding window) ardışık kare oylaması.
- **Raporlama Modülü:** Sürüş sonu istatistiklerini ve güvenlik skorunu otomatik kurumsal PDF karnesine dönüştürme.

## Kurulum ve Çalıştırma

1. Sanal ortamı oluşturun ve aktif edin:
   ```bash
   python -m venv venv
   # Windows için:
   .\venv\Scripts\activate
   ```

2. Gerekli kütüphaneleri kurun:
   ```bash
   pip install -r requirements.txt
   ```

3. Ana uygulamayı başlatın:
   ```bash
   python main.py
   ```

## Mimari Özeti
- **UI Layer (`ui/`):** PyQt5 tabanlı GUI ve video işleme thread'leri.
- **Services Layer (`services/`):** Sesli uyarı (TTS) ve raporlama (PDF) mantığı.
- **Utils Layer (`utils/`):** Kamera yönetimi, metrik hesaplama ve ML model yükleme yardımcıları.
- **Main Entry:** `main.py` ve `app_launcher.py` ile uygulama başlatma.

## Kullanılan Modeller
- **Drowsiness Detection:** MediaPipe Face Mesh & Holistic.
- **Distraction Detection (Phone):** YOLOv8n (Önceden eğitilmiş).

## Güvenlik ve Performans
Sistem, yanlış pozitifleri minimuma indirmek için **ardışık kare analizi** (consecutive frame analysis) kullanır. 5 saniyeden fazla süren tehlikeli durumlar, sürücüyü uyarmadan önce birikimli risk eşiğini geçmelidir.

## Loglama
Sistem çıktıları konsola yazdırılır ve `output/logs/` klasörüne zaman damgalı log dosyaları kaydedilir.