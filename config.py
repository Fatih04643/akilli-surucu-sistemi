"""
Akıllı Sürücü Yorgunluk ve Dikkat Dağınıklığı Tespit Sistemi - Konfigürasyon Modülü.

Bu modül, sistemin tüm eşik değerlerini, MediaPipe Face Mesh landmark
indekslerini ve genel ayarları merkezi bir noktadan yönetir.
"""

from typing import Final

# =============================================================================
#  Göz Kapalılık Tespiti (EAR - Eye Aspect Ratio)
# =============================================================================

EAR_THRESHOLD: Final[float] = 0.25
"""Göz kapalı sayılması için EAR değerinin altında kalması gereken eşik.
Değer düştükçe gözün daha fazla kapanması gerekir."""

CONSECUTIVE_FRAMES_DROWSY: Final[int] = 20
"""Sürücünün 'uykulu' olarak sınıflandırılması için EAR eşiğinin altında
kalması gereken ardışık kare (frame) sayısı."""

# =============================================================================
#  Esneme Tespiti (MAR - Mouth Aspect Ratio)
# =============================================================================

MAR_THRESHOLD: Final[float] = 0.65
"""Ağız açık (esneme) sayılması için MAR değerinin üstünde kalması gereken eşik."""

CONSECUTIVE_FRAMES_YAWN: Final[int] = 15
"""Esneme olarak sınıflandırılması için MAR eşiğinin üstünde kalması gereken
ardışık kare (frame) sayısı."""

# =============================================================================
#  Kafa Pozisyonu Tespiti (Head Pose Estimation)
# =============================================================================

HEAD_YAW_THRESHOLD: Final[float] = 30.0
"""Kafa sağa veya sola dönüş açısı sınırı (derece). Bu değerin üzerindeki
yaw açıları dikkat dağınıklığı olarak değerlendirilir."""

HEAD_PITCH_THRESHOLD: Final[float] = 20.0
"""Kafa yukarı veya aşağı eğilme açısı sınırı (derece). Bu değerin üzerindeki
pitch açıları dikkat dağınıklığı olarak değerlendirilir."""

# =============================================================================
#  MediaPipe Face Mesh Landmark İndeksleri
# =============================================================================

LEFT_EYE_INDICES: Final[list[int]] = [362, 385, 387, 263, 373, 380]
"""Sol göz çevresindeki 6 referans noktasının MediaPipe Face Mesh indeksleri.

Sıralama (EAR hesabı için):
    [0] P1 - Sol köşe (dış kanto)
    [1] P2 - Üst göz kapağı (dış)
    [2] P3 - Üst göz kapağı (iç)
    [3] P4 - Sağ köşe (iç kanto)
    [4] P5 - Alt göz kapağı (iç)
    [5] P6 - Alt göz kapağı (dış)
"""

RIGHT_EYE_INDICES: Final[list[int]] = [33, 160, 158, 133, 153, 144]
"""Sağ göz çevresindeki 6 referans noktasının MediaPipe Face Mesh indeksleri.

Sıralama (EAR hesabı için):
    [0] P1 - Sol köşe (iç kanto)
    [1] P2 - Üst göz kapağı (iç)
    [2] P3 - Üst göz kapağı (dış)
    [3] P4 - Sağ köşe (dış kanto)
    [4] P5 - Alt göz kapağı (dış)
    [5] P6 - Alt göz kapağı (iç)
"""

MOUTH_INDICES: Final[list[int]] = [78, 13, 311, 308, 402, 14]
"""Ağız / dudak çevresindeki 6 referans noktasının MediaPipe Face Mesh indeksleri.

Sıralama (MAR hesabı için):
    [0] P1 - Sol ağız köşesi
    [1] P2 - Üst dudak ortası (üst)
    [2] P3 - Sağ ağız köşesi
    [3] P4 - Sağ ağız köşesi (alt kenar)
    [4] P5 - Alt dudak ortası (alt)
    [5] P6 - Üst dudak ortası (alt)
"""

# =============================================================================
#  Ön İşleme (Preprocessing) Ayarları
# =============================================================================

CLAHE_CLIP_LIMIT: Final[float] = 2.0
"""CLAHE algoritmasının kontrast kırpma limiti. Yüksek değerler daha fazla
kontrast artışı sağlar ancak gürültüyü de artırabilir."""

CLAHE_TILE_GRID_SIZE: Final[tuple[int, int]] = (8, 8)
"""CLAHE algoritmasının görüntüyü böldüğü ızgara boyutu (satır, sütun)."""

GAUSSIAN_KERNEL_SIZE: Final[tuple[int, int]] = (3, 3)
"""Gürültü azaltma için kullanılan Gaussian Blur çekirdek boyutu.
Tek sayı çiftleri olmalıdır (3×3, 5×5 vb.)."""

# =============================================================================
#  Kafa Pozisyonu - 3B Model Noktaları
# =============================================================================

HEAD_POSE_3D_MODEL_POINTS_INDICES: Final[list[int]] = [1, 152, 33, 263, 61, 291]
"""solvePnP için kullanılan 6 temel yüz landmark indeksi.

Sıralama:
    [0] Burun ucu       (indeks 1)
    [1] Çene            (indeks 152)
    [2] Sol göz köşesi  (indeks 33)
    [3] Sağ göz köşesi  (indeks 263)
    [4] Sol ağız köşesi (indeks 61)
    [5] Sağ ağız köşesi (indeks 291)
"""
