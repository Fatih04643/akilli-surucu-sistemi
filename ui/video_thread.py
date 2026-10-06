"""
Video İşleme Thread Modülü.

Web kamerasını arka plan thread'inde okuyarak her kare üzerinde:
- CLAHE ön işleme
- MediaPipe FaceLandmarker (Tasks API) ile yüz tespiti
- EAR, MAR ve kafa pozisyonu hesaplama
- Karar motoru ile durum değerlendirmesi
- Görüntü üzerine çizim (yüz ağı, konturlar, 3B eksen)

işlemlerini gerçekleştirir ve sonuçları PyQt sinyalleri ile UI thread'ine
iletir.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker
from PyQt5.QtGui import QImage

# MediaPipe Tasks API (≥ 0.10.20)
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.face_landmarker import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    FaceLandmarkerResult,
    FaceLandmarksConnections,
)
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)
from mediapipe.tasks.python.vision import drawing_utils as mp_draw
from mediapipe.tasks.python.vision import drawing_styles as mp_styles
from mediapipe import Image as MpImage, ImageFormat as MpImageFormat

import config
from core.preprocessor import FramePreprocessor
from core.metrics import calculate_ear, calculate_mar, estimate_head_pose, calculate_iris_center, estimate_gaze_direction
from core.decision_engine import DecisionEngine
from core.detector import PhoneDetector
from services.report_service import ReportService

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Model Dosya Yolu
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_DIR: Final[Path] = Path(__file__).resolve().parent.parent / "models"
_FACE_LANDMARKER_MODEL: Final[Path] = _MODEL_DIR / "face_landmarker.task"

# ─────────────────────────────────────────────────────────────────────────────
#  Çizim Sabitleri
# ─────────────────────────────────────────────────────────────────────────────

# Göz ve ağız kontur indeksleri (iç kontur çizimi için)
_LEFT_EYE_CONTOUR: Final[list[int]] = [
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
]
_RIGHT_EYE_CONTOUR: Final[list[int]] = [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
]
_MOUTH_CONTOUR: Final[list[int]] = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78,
]

# 3B eksen çizimi için renkler (BGR)
_AXIS_COLORS: Final[dict[str, tuple[int, int, int]]] = {
    "x": (0, 0, 255),   # Kırmızı
    "y": (0, 255, 0),   # Yeşil
    "z": (255, 0, 0),   # Mavi
}

# 3B model noktaları (head pose – metrics.py ile aynı)
_3D_MODEL_POINTS: NDArray[np.float64] = np.array([
    (0.0,      0.0,       0.0),
    (0.0,     -330.0,    -65.0),
    (-225.0,   170.0,    -135.0),
    (225.0,    170.0,    -135.0),
    (-150.0,  -150.0,    -125.0),
    (150.0,   -150.0,    -125.0),
], dtype=np.float64)


class VideoThread(QThread):
    """Arka plan kamera okuma ve görüntü işleme thread'i.

    Sinyaller:
        frame_ready(QImage): Üzerine çizim yapılmış kareyi gönderir.
        metrics_ready(float, float, float, float): (EAR, MAR, Yaw, Pitch).
        status_ready(str, str, int): (durum_metni, hex_renk, yorgunluk_skoru).
    """

    # ── PyQt Sinyalleri ──
    frame_ready = pyqtSignal(QImage)
    metrics_ready = pyqtSignal(float, float, float, float)
    status_ready = pyqtSignal(str, str, int)
    gaze_ready = pyqtSignal(str)  # Yeni sinyal

    def __init__(
        self,
        camera_index: int = 0,
        decision_engine: DecisionEngine | None = None,
        report_service: ReportService | None = None,
        parent=None,
    ) -> None:
        """VideoThread nesnesini başlatır.

        Args:
            camera_index: cv2.VideoCapture kamera indeksi (varsayılan 0).
            decision_engine: Karar motoru nesnesi. ``None`` ise dahili bir
                ``DecisionEngine`` oluşturulur.
            parent: Üst QObject.
        """
        super().__init__(parent)

        self._camera_index: int = camera_index
        self._running: bool = False
        self._mutex: QMutex = QMutex()

        # Core bileşenler
        self._preprocessor = FramePreprocessor()
        self._decision_engine = decision_engine or DecisionEngine()
        self._phone_detector = PhoneDetector()
        self._report_service = report_service

        self._face_landmarker: FaceLandmarker | None = None
        self._cap: cv2.VideoCapture | None = None

    # ─────────────────────────────────────────────────────────────────────
    #  FaceLandmarker Başlatma
    # ─────────────────────────────────────────────────────────────────────

    def _init_face_landmarker(self) -> FaceLandmarker:
        """MediaPipe FaceLandmarker nesnesini oluşturur."""
        model_path = str(_FACE_LANDMARKER_MODEL)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"FaceLandmarker model dosyası bulunamadı: {model_path}\n"
                "Modeli indirmek için:\n"
                "  wget https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
            )

        # C++ backend'in Türkçe karakterli dizinlerde çökmesini önlemek için
        # modeli byte olarak okuyup buffer olarak veriyoruz.
        with open(model_path, "rb") as f:
            model_data = f.read()

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model_data),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        return FaceLandmarker.create_from_options(options)

    # ─────────────────────────────────────────────────────────────────────
    #  Yaşam Döngüsü
    # ─────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Thread ana döngüsü – kamera okuma ve işleme."""
        # FaceLandmarker'ı thread içinde başlat (thread-safety)
        try:
            self._face_landmarker = self._init_face_landmarker()
        except FileNotFoundError as e:
            logger.error(str(e))
            return

        # Windows ortamında hızlı başlatma için CAP_DSHOW ekleniyor
        self._cap = cv2.VideoCapture(self._camera_index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            logger.error("Kamera açılamadı (index=%d)", self._camera_index)
            return

        self._running = True
        frame_timestamp_ms = 0
        logger.info("VideoThread başlatıldı (kamera %d).", self._camera_index)

        while True:
            with QMutexLocker(self._mutex):
                if not self._running:
                    break

            ret, raw_frame = self._cap.read()
            if not ret:
                logger.warning("Kare okunamadı, döngü sonlandırılıyor.")
                break

            frame_timestamp_ms += 33  # ~30 fps simülasyonu

            try:
                display_frame, ear, mar, yaw, pitch, is_phone_used, gaze_state = self._process_frame(
                    raw_frame, frame_timestamp_ms,
                )
            except Exception:
                logger.exception("Kare işleme hatası.")
                display_frame = raw_frame
                ear, mar, yaw, pitch, is_phone_used, gaze_state = 0.0, 0.0, 0.0, 0.0, False, "MERKEZ"

            # Karar motoru değerlendirmesi
            result = self._decision_engine.evaluate(
                ear=ear, mar=mar, yaw=yaw, pitch=pitch, is_phone_used=is_phone_used, gaze_state=gaze_state
            )

            # Rapor servisi güncellemesi
            if self._report_service:
                is_gaze_off_road = (gaze_state != "MERKEZ")
                self._report_service.update_telemetry(result.status.name, is_gaze_off_road)
                if result.status.name != "NORMAL" and result.status.name != getattr(self, '_last_status', "NORMAL"):
                    self._report_service.record_violation(result.status.name)
                self._last_status = result.status.name

            # ── Sinyalleri gönder ──
            qt_image = self._frame_to_qimage(display_frame)
            self.frame_ready.emit(qt_image)
            self.metrics_ready.emit(ear, mar, yaw, pitch)
            self.status_ready.emit(result.label, result.color, result.fatigue_score)
            self.gaze_ready.emit(gaze_state)

        # Temizlik
        self._cleanup()
        logger.info("VideoThread sonlandırıldı.")

    def stop(self) -> None:
        """Thread'i güvenli şekilde durdurur."""
        with QMutexLocker(self._mutex):
            self._running = False

    # ─────────────────────────────────────────────────────────────────────
    #  Kare İşleme
    # ─────────────────────────────────────────────────────────────────────

    def _process_frame(
        self,
        frame: NDArray[np.uint8],
        timestamp_ms: int,
    ) -> tuple[NDArray[np.uint8], float, float, float, float, bool, str]:
        """Tek bir kare üzerinde tüm işleme pipeline'ını çalıştırır.

        Returns:
        """
        h, w, _ = frame.shape

        # 1. Ön işleme (CLAHE + Gaussian)
        enhanced = self._preprocessor.enhance(frame)

        # 2. MediaPipe FaceLandmarker (RGB dönüşümü gerekli)
        rgb_frame = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        mp_image = MpImage(image_format=MpImageFormat.SRGB, data=rgb_frame)

        result: FaceLandmarkerResult = self._face_landmarker.detect_for_video(
            mp_image, timestamp_ms,
        )

        ear, mar, yaw, pitch = 0.0, 0.0, 0.0, 0.0
        gaze_state = "MERKEZ"
        display = enhanced.copy()

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]  # İlk yüz

            # Tüm landmark'ları piksel koordinatlarına dönüştür
            pts = [(lm.x * w, lm.y * h) for lm in landmarks]

            # ── EAR hesaplama ──
            left_eye = [pts[i] for i in config.LEFT_EYE_INDICES]
            right_eye = [pts[i] for i in config.RIGHT_EYE_INDICES]
            ear_left = calculate_ear(left_eye)
            ear_right = calculate_ear(right_eye)
            ear = (ear_left + ear_right) / 2.0

            # ── MAR hesaplama ──
            mouth = [pts[i] for i in config.MOUTH_INDICES]
            mar = calculate_mar(mouth)

            # ── Kafa pozisyonu ──
            head_pts = [pts[i] for i in config.HEAD_POSE_3D_MODEL_POINTS_INDICES]
            try:
                yaw, pitch, _ = estimate_head_pose(head_pts, (h, w, 3))
            except ValueError:
                yaw, pitch = 0.0, 0.0

            # ── Iris & Gaze ──
            # MediaPipe'ta genellikle sol göz: 468-472, sağ göz: 473-477
            # Ama refine_landmarks kapatıldıysa veya farklıysa list out of range olabilir
            try:
                left_iris_pts = [pts[i] for i in range(468, 473)]
                right_iris_pts = [pts[i] for i in range(473, 478)]
                
                left_center = calculate_iris_center(left_iris_pts)
                right_center = calculate_iris_center(right_iris_pts)
                
                # Göz dış/iç köşeleri (MediaPipe)
                # Sol göz (ekranda sağda) dış: 263, iç: 362
                # Sağ göz (ekranda solda) dış: 33, iç: 133
                left_eye_inner, left_eye_outer = pts[362], pts[263]
                right_eye_inner, right_eye_outer = pts[133], pts[33]
                
                left_gaze = estimate_gaze_direction(left_center, left_eye_inner, left_eye_outer)
                right_gaze = estimate_gaze_direction(right_center, right_eye_inner, right_eye_outer)
                
                # Genel bakış yönü (basit çoğunluk)
                if left_gaze == right_gaze:
                    # 'İÇ' veya 'DIŞ' olabilir. 
                    # Sol göz için 'DIŞ' (263'e yakın) = SOLA BAKIŞ
                    # Sağ göz için 'DIŞ' (33'e yakın) = SAĞA BAKIŞ
                    # estimate_gaze_direction aslında oran ile sadece iç/dış diyor.
                    # Oranla daha iyi bir hesaplama yapalım:
                    pass
                
                left_x_min = min(left_eye_inner[0], left_eye_outer[0])
                left_x_max = max(left_eye_inner[0], left_eye_outer[0])
                left_ratio = (left_center[0] - left_x_min) / (left_x_max - left_x_min + 1e-6)
                
                right_x_min = min(right_eye_inner[0], right_eye_outer[0])
                right_x_max = max(right_eye_inner[0], right_eye_outer[0])
                right_ratio = (right_center[0] - right_x_min) / (right_x_max - right_x_min + 1e-6)
                
                avg_ratio = (left_ratio + right_ratio) / 2.0
                if avg_ratio < 0.38:
                    gaze_state = "SAĞ" # Ekranda sola yakındır (sürücünün sağı)
                elif avg_ratio > 0.62:
                    gaze_state = "SOL" # Ekranda sağa yakındır (sürücünün solu)
                else:
                    gaze_state = "MERKEZ"
                
                # İris merkezlerine hedef noktası çizimi
                cv2.circle(display, (int(left_center[0]), int(left_center[1])), 2, (0, 0, 255), -1)
                cv2.circle(display, (int(right_center[0]), int(right_center[1])), 2, (0, 0, 255), -1)
                
            except IndexError:
                # İris landmarkları yoksa (refine_landmarks=False)
                gaze_state = "MERKEZ"

            # ── Görüntü üzerine çizim ──
            self._draw_face_mesh(display, landmarks)
            self._draw_contour(display, pts, _LEFT_EYE_CONTOUR, (0, 255, 200))
            self._draw_contour(display, pts, _RIGHT_EYE_CONTOUR, (0, 255, 200))
            self._draw_contour(display, pts, _MOUTH_CONTOUR, (0, 200, 255))
            self._draw_head_pose_axes(display, head_pts, (h, w, 3))

        else:
            # Yüz bulunamadığında karar motorunu sıfırla
            self._decision_engine.reset()

        # 3. Telefon Tespiti
        is_phone_used = self._phone_detector.detect(display)

        return display, ear, mar, yaw, pitch, is_phone_used, gaze_state

    # ─────────────────────────────────────────────────────────────────────
    #  Çizim Yardımcıları
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _draw_face_mesh(frame: NDArray[np.uint8], landmarks) -> None:
        """Yüz ağı tesselasyon çizgilerini çizer (Tasks API)."""
        mp_draw.draw_landmarks(
            image=frame,
            landmark_list=landmarks,
            connections=FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_draw.DrawingSpec(
                color=(80, 110, 10), thickness=1, circle_radius=1,
            ),
            is_drawing_landmarks=False,
        )

    @staticmethod
    def _draw_contour(
        frame: NDArray[np.uint8],
        pts: list[tuple[float, float]],
        indices: list[int],
        color: tuple[int, int, int],
    ) -> None:
        """Verilen indeks listesinden kontur çizer."""
        contour = np.array(
            [(int(pts[i][0]), int(pts[i][1])) for i in indices],
            dtype=np.int32,
        )
        cv2.polylines(frame, [contour], isClosed=True, color=color, thickness=1)

    @staticmethod
    def _draw_head_pose_axes(
        frame: NDArray[np.uint8],
        head_pts: list[tuple[float, float]],
        frame_shape: tuple[int, ...],
    ) -> None:
        """Burun ucundan başlayan 3B eksen çizgilerini çizer.

        Kafa yönünü görsel olarak göstermek için X (kırmızı), Y (yeşil),
        Z (mavi) eksenleri çizilir.
        """
        h, w = frame_shape[0], frame_shape[1]

        image_points = np.array(head_pts, dtype=np.float64)
        focal_length = float(w)
        camera_matrix = np.array([
            [focal_length, 0.0,          w / 2.0],
            [0.0,          focal_length, h / 2.0],
            [0.0,          0.0,          1.0],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            _3D_MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return

        # 3B eksen uç noktaları (mm cinsinden)
        axis_length = 150.0
        axis_3d = np.array([
            [axis_length, 0.0,          0.0],
            [0.0,         axis_length,   0.0],
            [0.0,         0.0,          -axis_length],
        ], dtype=np.float64)

        projected, _ = cv2.projectPoints(
            axis_3d, rvec, tvec, camera_matrix, dist_coeffs,
        )

        nose = (int(head_pts[0][0]), int(head_pts[0][1]))

        for i, axis_key in enumerate(("x", "y", "z")):
            end = (int(projected[i][0][0]), int(projected[i][0][1]))
            cv2.line(frame, nose, end, _AXIS_COLORS[axis_key], 2)

    # ─────────────────────────────────────────────────────────────────────
    #  Yardımcı
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _frame_to_qimage(frame: NDArray[np.uint8]) -> QImage:
        """BGR numpy dizisini QImage'e dönüştürür."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        return QImage(
            rgb.data, w, h, bytes_per_line, QImage.Format_RGB888,
        ).copy()  # .copy() – numpy bellek güvenliği

    def _cleanup(self) -> None:
        """Tüm kaynakları güvenli şekilde serbest bırakır."""
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
            logger.debug("Kamera serbest bırakıldı.")
        self._cap = None

        if self._face_landmarker is not None:
            self._face_landmarker.close()
            self._face_landmarker = None
            logger.debug("FaceLandmarker kapatıldı.")

    def __del__(self) -> None:
        """Yıkıcı."""
        pass
