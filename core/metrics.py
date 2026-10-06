"""
Metrik Hesaplama Modülü.

Sürücü yorgunluk ve dikkat dağınıklığı tespiti için gerekli üç temel
metriği hesaplar:

- **EAR** (Eye Aspect Ratio) : Göz açıklık oranı
- **MAR** (Mouth Aspect Ratio): Ağız açıklık oranı (esneme tespiti)
- **Head Pose** (Kafa Pozisyonu): Euler açıları (Yaw, Pitch, Roll)

Tipik Kullanım:
    >>> from core.metrics import calculate_ear, calculate_mar, estimate_head_pose
    >>>
    >>> ear = calculate_ear(eye_landmarks)
    >>> mar = calculate_mar(mouth_landmarks)
    >>> yaw, pitch, roll = estimate_head_pose(all_landmarks, (480, 640))
"""

from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray


# =============================================================================
#  Yardımcı Fonksiyonlar
# =============================================================================

def _euclidean_distance(
    point_a: Sequence[float],
    point_b: Sequence[float],
) -> float:
    """İki nokta arasındaki Öklid mesafesini hesaplar.

    Args:
        point_a: İlk noktanın ``(x, y)`` koordinatları.
        point_b: İkinci noktanın ``(x, y)`` koordinatları.

    Returns:
        İki nokta arasındaki Öklid mesafesi.

    Example:
        >>> _euclidean_distance((0, 0), (3, 4))
        5.0
    """
    return math.sqrt(
        (point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2
    )


# =============================================================================
#  EAR (Eye Aspect Ratio) Hesaplama
# =============================================================================

def calculate_ear(
    eye_landmarks: Sequence[Sequence[float]],
) -> float:
    """Göz Açıklık Oranı (Eye Aspect Ratio - EAR) hesaplar.

    Solanki et al. (2016) tarafından önerilen formül kullanılır::

                 ||P2 - P6|| + ||P3 - P5||
        EAR  =  ─────────────────────────────
                       2 × ||P1 - P4||

    Burada P1..P6, göz çevresindeki 6 referans noktasıdır:
        - P1, P4 : Göz köşeleri (yatay eksen)
        - P2, P6 : Üst ve alt göz kapağı (dış dikey eksen)
        - P3, P5 : Üst ve alt göz kapağı (iç dikey eksen)

    Args:
        eye_landmarks: 6 adet ``(x, y)`` koordinat çiftinden oluşan dizi.
            Sıralama: ``[P1, P2, P3, P4, P5, P6]``

    Returns:
        EAR değeri. Göz açıkken ~0.3, kapalıyken ~0.0 civarındadır.

    Raises:
        ValueError: Landmark sayısı 6 değilse.

    Example:
        >>> # Tam açık göz simülasyonu
        >>> landmarks = [(0, 0), (1, 2), (3, 2), (4, 0), (3, -2), (1, -2)]
        >>> ear = calculate_ear(landmarks)
        >>> ear > 0.0
        True
    """
    if len(eye_landmarks) != 6:
        raise ValueError(
            f"EAR hesabı için 6 landmark gerekli, verilen: {len(eye_landmarks)}"
        )

    # Dikey mesafeler
    vertical_1: float = _euclidean_distance(eye_landmarks[1], eye_landmarks[5])
    vertical_2: float = _euclidean_distance(eye_landmarks[2], eye_landmarks[4])

    # Yatay mesafe
    horizontal: float = _euclidean_distance(eye_landmarks[0], eye_landmarks[3])

    # Sıfıra bölme koruması
    if horizontal < 1e-6:
        return 0.0

    ear: float = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


# =============================================================================
#  MAR (Mouth Aspect Ratio) Hesaplama
# =============================================================================

def calculate_mar(
    mouth_landmarks: Sequence[Sequence[float]],
) -> float:
    """Ağız Açıklık Oranı (Mouth Aspect Ratio - MAR) hesaplar.

    EAR formülüne benzer bir yaklaşım kullanılır::

                 ||P2 - P6|| + ||P3 - P5||
        MAR  =  ─────────────────────────────
                       2 × ||P1 - P4||

    Burada:
        - P1 (indeks 78)  : Sol ağız köşesi
        - P2 (indeks 13)  : Üst dudak ortası (üst)
        - P3 (indeks 311) : Sağ ağız köşesi üst
        - P4 (indeks 308) : Sağ ağız köşesi alt
        - P5 (indeks 402) : Alt dudak ortası (alt)
        - P6 (indeks 14)  : Üst dudak ortası (alt)

    Args:
        mouth_landmarks: 6 adet ``(x, y)`` koordinat çiftinden oluşan dizi.
            Sıralama: ``[P1, P2, P3, P4, P5, P6]``

    Returns:
        MAR değeri. Ağız kapalıyken düşük, esneme sırasında yüksek.

    Raises:
        ValueError: Landmark sayısı 6 değilse.

    Example:
        >>> landmarks = [(0, 0), (1, 3), (2, 3), (3, 0), (2, -3), (1, -3)]
        >>> mar = calculate_mar(landmarks)
        >>> mar > 0.0
        True
    """
    if len(mouth_landmarks) != 6:
        raise ValueError(
            f"MAR hesabı için 6 landmark gerekli, verilen: {len(mouth_landmarks)}"
        )

    # Dikey mesafeler
    vertical_1: float = _euclidean_distance(
        mouth_landmarks[1], mouth_landmarks[5]
    )
    vertical_2: float = _euclidean_distance(
        mouth_landmarks[2], mouth_landmarks[4]
    )

    # Yatay mesafe
    horizontal: float = _euclidean_distance(
        mouth_landmarks[0], mouth_landmarks[3]
    )

    # Sıfıra bölme koruması
    if horizontal < 1e-6:
        return 0.0

    mar: float = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return mar


# =============================================================================
#  Kafa Pozisyonu Tahmini (Head Pose Estimation)
# =============================================================================

# Jenerik 3B kafa modeli koordinatları (mm cinsinden).
# Bu noktalar, ortalama bir insan yüzünün 6 temel noktasına karşılık gelir.
_3D_MODEL_POINTS: NDArray[np.float64] = np.array([
    (0.0,      0.0,       0.0),      # Burun ucu
    (0.0,     -330.0,    -65.0),      # Çene
    (-225.0,   170.0,    -135.0),     # Sol göz sol köşesi
    (225.0,    170.0,    -135.0),     # Sağ göz sağ köşesi
    (-150.0,  -150.0,    -125.0),     # Sol ağız köşesi
    (150.0,   -150.0,    -125.0),     # Sağ ağız köşesi
], dtype=np.float64)


def estimate_head_pose(
    landmarks: Sequence[Sequence[float]],
    frame_shape: tuple[int, ...],
) -> tuple[float, float, float]:
    """Kafa pozisyonunu tahmin ederek Euler açılarını döndürür.

    2B yüz landmark'ları ile jenerik bir 3B kafa modeli arasında
    ``cv2.solvePnP`` kullanarak dönme vektörünü (``rvec``) hesaplar,
    ardından bu vektörü dönüş matrisine ve Euler açılarına çevirir.

    Kullanılan 6 Landmark (MediaPipe Face Mesh indeksleri):
        - **Burun ucu**       → indeks 1
        - **Çene**            → indeks 152
        - **Sol göz köşesi**  → indeks 33
        - **Sağ göz köşesi**  → indeks 263
        - **Sol ağız köşesi** → indeks 61
        - **Sağ ağız köşesi** → indeks 291

    Args:
        landmarks: 6 adet ``(x, y)`` piksel koordinat çiftinden oluşan dizi.
            Sıralama yukarıdaki listeye uygun olmalıdır.
        frame_shape: Görüntü boyutları ``(yükseklik, genişlik, ...)``.
            Kamera iç parametrelerini (intrinsics) yaklaşık olarak
            hesaplamak için kullanılır.

    Returns:
        ``(yaw, pitch, roll)`` Euler açıları (derece cinsinden).

        - **Yaw**   : Sağa/sola dönüş. Pozitif → sola, Negatif → sağa.
        - **Pitch** : Yukarı/aşağı eğilme. Pozitif → aşağı, Negatif → yukarı.
        - **Roll**  : Yana yatma. Pozitif → saat yönü, Negatif → saat yönü tersi.

    Raises:
        ValueError: Landmark sayısı 6 değilse veya ``solvePnP`` çözüm bulamazsa.

    Example:
        >>> # Merkezde duran bir yüz simülasyonu
        >>> center_landmarks = [
        ...     (320, 240), (320, 400), (250, 190),
        ...     (390, 190), (280, 350), (360, 350),
        ... ]
        >>> yaw, pitch, roll = estimate_head_pose(center_landmarks, (480, 640, 3))
        >>> abs(yaw) < 45  # Makul bir aralıkta olmalı
        True
    """
    if len(landmarks) != 6:
        raise ValueError(
            f"Kafa pozisyonu tahmini için 6 landmark gerekli, "
            f"verilen: {len(landmarks)}"
        )

    # 2B landmark'ları numpy dizisine dönüştür
    image_points: NDArray[np.float64] = np.array(
        landmarks, dtype=np.float64,
    )

    # Görüntü boyutlarından yaklaşık kamera iç parametrelerini oluştur
    height: int = frame_shape[0]
    width: int = frame_shape[1]

    # Odak uzaklığını görüntü genişliğine göre yaklaşık hesapla
    focal_length: float = float(width)

    # Optik merkez (görüntünün ortası)
    center_x: float = width / 2.0
    center_y: float = height / 2.0

    # Kamera matrisi (intrinsic matrix)
    camera_matrix: NDArray[np.float64] = np.array([
        [focal_length,  0.0,          center_x],
        [0.0,           focal_length, center_y],
        [0.0,           0.0,          1.0],
    ], dtype=np.float64)

    # Distorsiyon katsayıları (basitleştirilmiş: distorsiyon yok)
    dist_coeffs: NDArray[np.float64] = np.zeros((4, 1), dtype=np.float64)

    # PnP çözümü (Perspective-n-Point)
    success, rvec, tvec = cv2.solvePnP(
        objectPoints=_3D_MODEL_POINTS,
        imagePoints=image_points,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        raise ValueError(
            "cv2.solvePnP çözüm bulamadı. Landmark koordinatlarını "
            "kontrol ediniz."
        )

    # Dönme vektöründen (rvec) dönme matrisine (rotation matrix) çevir
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    # Euler açılarını hesapla (dönme matrisinden)
    yaw, pitch, roll = _rotation_matrix_to_euler_angles(rotation_matrix)

    return yaw, pitch, roll


def _rotation_matrix_to_euler_angles(
    rotation_matrix: NDArray[np.float64],
) -> tuple[float, float, float]:
    """3×3 dönme matrisinden Euler açılarını (Yaw, Pitch, Roll) hesaplar.

    Dönme matrisi ``R = Rz(yaw) × Ry(pitch) × Rx(roll)`` şeklinde
    ayrıştırılır. Gimbal lock durumunda pitch ±90° olarak sabitlenir.

    Args:
        rotation_matrix: 3×3 ortogonal dönme matrisi.

    Returns:
        ``(yaw, pitch, roll)`` derece cinsinden Euler açıları.
    """
    # Gimbal lock kontrolü: R[2,0] = -sin(pitch)
    sy: float = math.sqrt(
        rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2
    )

    is_singular: bool = sy < 1e-6

    if not is_singular:
        roll = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        pitch = math.atan2(-rotation_matrix[2, 0], sy)
        yaw = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        # Gimbal lock: pitch ≈ ±90°
        roll = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        pitch = math.atan2(-rotation_matrix[2, 0], sy)
        yaw = 0.0

    # Radyandan dereceye çevir
    yaw_deg: float = math.degrees(yaw)
    pitch_deg: float = math.degrees(pitch)
    roll_deg: float = math.degrees(roll)

    return yaw_deg, pitch_deg, roll_deg


# =============================================================================
#  Göz Bebeği (Iris) ve Bakış Yönü (Gaze) Tahmini
# =============================================================================

def calculate_iris_center(iris_landmarks: Sequence[Sequence[float]]) -> tuple[float, float]:
    """İris landmarklarının merkez noktasını hesaplar.
    
    Args:
        iris_landmarks: Göz bebeği için noktalar (örneğin sol göz için 468-472)
    Returns:
        (x, y) merkez koordinatı
    """
    if not iris_landmarks:
        return 0.0, 0.0
    
    x_coords = [p[0] for p in iris_landmarks]
    y_coords = [p[1] for p in iris_landmarks]
    
    return sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords)

def estimate_gaze_direction(
    iris_center: tuple[float, float],
    eye_inner: Sequence[float],
    eye_outer: Sequence[float]
) -> str:
    """Göz bebeğinin yatay pozisyonuna göre bakış yönünü hesaplar.
    
    Args:
        iris_center: (x, y) iris merkez noktası.
        eye_inner: (x, y) gözün iç köşesi.
        eye_outer: (x, y) gözün dış köşesi.
    
    Returns:
        "MERKEZ", "SOL" veya "SAĞ"
    """
    # X koordinatları arasındaki mesafeler
    eye_width = abs(eye_outer[0] - eye_inner[0])
    if eye_width < 1e-6:
        return "MERKEZ"
        
    # İrisin iç köşeye uzaklığı
    iris_to_inner = abs(iris_center[0] - eye_inner[0])
    
    # Oran: 0.0 (tamamen iç köşeye bakıyor), 1.0 (tamamen dış köşeye bakıyor)
    ratio = iris_to_inner / eye_width
    
    # Eşik değerleri (yaklaşık)
    # Bu oran kamera açısı ve kişiye göre değişebilir, basit bir heuristik kullanıyoruz.
    if ratio < 0.40:
        # İris iç köşeye daha yakın
        # Sol göz için sağa bakış, sağ göz için sola bakış olabilir.
        # Biz doğrudan x koordinatlarına göre genel bir yön çıkaracağız.
        return "İÇ"
    elif ratio > 0.60:
        # İris dış köşeye daha yakın
        return "DIŞ"
    else:
        return "MERKEZ"
