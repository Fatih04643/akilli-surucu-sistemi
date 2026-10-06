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