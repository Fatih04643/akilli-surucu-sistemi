"""
Görüntü Ön İşleme Modülü.

Gece sürüşü ve düşük ışık koşullarında kamera görüntüsünün kalitesini
artırmak için CLAHE tabanlı kontrast iyileştirmesi ve Gaussian Blur
tabanlı gürültü azaltma işlemlerini uygular.

Tipik Kullanım:
    >>> import cv2
    >>> from core.preprocessor import FramePreprocessor
    >>>
    >>> preprocessor = FramePreprocessor()
    >>> frame = cv2.imread("test_frame.jpg")
    >>> enhanced = preprocessor.enhance(frame)
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

import config


class FramePreprocessor:
    """Gece ve düşük ışık koşulları için görüntü iyileştirme sınıfı.

    BGR formatındaki bir görüntüyü LAB renk uzayına çevirerek L (parlaklık)
    kanalına CLAHE uygular, ardından hafif bir Gaussian Blur ile sensör
    gürültüsünü temizler.

    Attributes:
        _clahe: OpenCV CLAHE nesnesi (kontrast iyileştirme).
        _gaussian_kernel: Gaussian Blur çekirdek boyutu.
    """

    def __init__(
        self,
        clip_limit: float = config.CLAHE_CLIP_LIMIT,
        tile_grid_size: tuple[int, int] = config.CLAHE_TILE_GRID_SIZE,
        gaussian_kernel: tuple[int, int] = config.GAUSSIAN_KERNEL_SIZE,
    ) -> None:
        """FramePreprocessor nesnesini başlatır.

        Args:
            clip_limit: CLAHE kontrast kırpma limiti. Varsayılan ``config.CLAHE_CLIP_LIMIT``.
            tile_grid_size: CLAHE ızgara boyutu ``(satır, sütun)``.
                Varsayılan ``config.CLAHE_TILE_GRID_SIZE``.
            gaussian_kernel: Gaussian Blur çekirdek boyutu ``(genişlik, yükseklik)``.
                Tek sayı çiftleri olmalıdır. Varsayılan ``config.GAUSSIAN_KERNEL_SIZE``.

        Raises:
            ValueError: ``gaussian_kernel`` boyutları çift sayı ise.
        """
        if gaussian_kernel[0] % 2 == 0 or gaussian_kernel[1] % 2 == 0:
            raise ValueError(
                f"Gaussian çekirdek boyutları tek sayı olmalıdır, "
                f"verilen: {gaussian_kernel}"
            )

        self._clahe: cv2.CLAHE = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size,
        )
        self._gaussian_kernel: tuple[int, int] = gaussian_kernel

    def enhance(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Görüntüye CLAHE kontrast iyileştirmesi ve gürültü azaltma uygular.

        İşlem Adımları:
            1. BGR → LAB renk uzayı dönüşümü.
            2. L (parlaklık) kanalına CLAHE uygulanması.
            3. LAB → BGR geri dönüşümü.
            4. Gaussian Blur ile sensör gürültüsünün azaltılması.

        Args:
            frame: BGR formatında 3 kanallı ``uint8`` görüntü dizisi.
                Boyut: ``(yükseklik, genişlik, 3)``.

        Returns:
            İyileştirilmiş BGR görüntü dizisi. Giriş ile aynı boyutta.

        Raises:
            ValueError: Giriş görüntüsü 3 kanallı değilse veya boşsa.

        Example:
            >>> import numpy as np
            >>> dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            >>> preprocessor = FramePreprocessor()
            >>> result = preprocessor.enhance(dummy)
            >>> result.shape
            (480, 640, 3)
        """
        if frame is None or frame.size == 0:
            raise ValueError("Giriş görüntüsü boş veya None.")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"3 kanallı BGR görüntü bekleniyor, "
                f"verilen boyut: {frame.shape}"
            )

        # 1. BGR → LAB dönüşümü
        lab: NDArray[np.uint8] = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        # 2. L kanalını ayır ve CLAHE uygula
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_enhanced: NDArray[np.uint8] = self._clahe.apply(l_channel)

        # 3. Kanalları birleştir ve LAB → BGR geri dönüşümü
        lab_enhanced: NDArray[np.uint8] = cv2.merge(
            [l_enhanced, a_channel, b_channel]
        )
        bgr_enhanced: NDArray[np.uint8] = cv2.cvtColor(
            lab_enhanced, cv2.COLOR_LAB2BGR
        )

        # 4. Gaussian Blur ile gürültü azaltma
        denoised: NDArray[np.uint8] = cv2.GaussianBlur(
            bgr_enhanced,
            self._gaussian_kernel,
            sigmaX=0,
        )

        return denoised

    def apply_clahe_only(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Yalnızca CLAHE kontrast iyileştirmesi uygular (blur olmadan).

        Gürültü azaltmanın istenilmediği veya ayrı bir adımda yapılacağı
        durumlar için kullanılır.

        Args:
            frame: BGR formatında 3 kanallı ``uint8`` görüntü dizisi.

        Returns:
            CLAHE uygulanmış BGR görüntü dizisi.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Giriş görüntüsü boş veya None.")

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_enhanced = self._clahe.apply(l_channel)
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])

        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    def apply_denoise_only(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Yalnızca Gaussian Blur gürültü azaltma uygular (CLAHE olmadan).

        Args:
            frame: BGR formatında 3 kanallı ``uint8`` görüntü dizisi.

        Returns:
            Gürültüsü azaltılmış BGR görüntü dizisi.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Giriş görüntüsü boş veya None.")

        return cv2.GaussianBlur(frame, self._gaussian_kernel, sigmaX=0)
