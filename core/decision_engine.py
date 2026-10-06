"""
Karar Motoru Modülü.

Anlık metrik değerlerini (EAR, MAR, Yaw, Pitch) yapılandırılmış eşik
değerleriyle karşılaştırarak sürücü durumunu belirler ve gerektiğinde
``AlertService`` üzerinden sesli uyarı tetikler.

Temel Özellikler:
    - **Oylama / Kararlılık Filtresi**: Yanlış alarmları önlemek için eşik
      ihlalinin ardışık kare sayısı kadar sürmesini zorunlu kılar.
    - **Öncelik Sıralaması**: ``DROWSY > YAWNING > DISTRACTED > NORMAL``
    - **Yorgunluk Skoru**: 0-100 arası normalize edilmiş skor.
    - **Renk Kodu**: Görsel geri bildirim için Hex renk değerleri.

Tipik Kullanım:
    >>> from core.decision_engine import DecisionEngine, DriverStatus
    >>> from services.alert_service import AlertService
    >>>
    >>> alert = AlertService()
    >>> alert.start()
    >>> engine = DecisionEngine(alert_service=alert)
    >>>
    >>> result = engine.evaluate(ear=0.18, mar=0.3, yaw=5.0, pitch=0.0)
    >>> print(result)
    {'status': <DriverStatus.NORMAL: 'NORMAL'>, ...}
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Final

import config

logger = logging.getLogger(__name__)


# =============================================================================
#  Sürücü Durumu Enum'u
# =============================================================================

class DriverStatus(enum.Enum):
    """Sürücünün anlık durumunu temsil eden sabit değerler.

    Her durum, bir metin etiketi ile ilişkilendirilmiştir:
        - ``NORMAL``     : Sürücü uyanık ve dikkatli.
        - ``DROWSY``     : Sürücü uykulu (göz kapanıklığı tespit edildi).
        - ``YAWNING``    : Sürücü esneme hareketinde (ağız açık).
        - ``DISTRACTED`` : Sürücü dikkat dağınıklığı gösteriyor (kafa dönük).
    """
    NORMAL     = "NORMAL"
    DROWSY     = "DROWSY"
    YAWNING    = "YAWNING"
    DISTRACTED = "DISTRACTED"
    PHONE_USAGE = "PHONE_USAGE"


# =============================================================================
#  Renk Kodları ve Sabitler
# =============================================================================

_STATUS_COLORS: Final[dict[DriverStatus, str]] = {
    DriverStatus.NORMAL:     "#00C853",   # Yeşil
    DriverStatus.DROWSY:     "#D50000",   # Kırmızı
    DriverStatus.YAWNING:    "#FFD600",   # Sarı
    DriverStatus.DISTRACTED: "#FF6D00",   # Turuncu
    DriverStatus.PHONE_USAGE: "#C62828",  # Koyu Kırmızı
}
"""Her durum için görsel arayüzde kullanılacak Hex renk kodları."""

_STATUS_LABELS: Final[dict[DriverStatus, str]] = {
    DriverStatus.NORMAL:     "Normal - Dikkatli",
    DriverStatus.DROWSY:     "UYARI: Uykulu!",
    DriverStatus.YAWNING:    "UYARI: Esneme Tespit Edildi!",
    DriverStatus.DISTRACTED: "UYARI: Dikkat Dağınıklığı!",
    DriverStatus.PHONE_USAGE: "KURAL İHLALİ: TELEFON KULLANIMI",
}
"""Her durum için kullanıcıya gösterilecek Türkçe metin etiketleri."""

_STATUS_ALERT_CATEGORY: Final[dict[DriverStatus, str]] = {
    DriverStatus.DROWSY:     "drowsy",
    DriverStatus.YAWNING:    "yawning",
    DriverStatus.DISTRACTED: "distracted",
    DriverStatus.PHONE_USAGE: "phone",
}
"""Uyarı gerektiren durumlar için AlertService kategori eşlemesi."""

_CONSECUTIVE_FRAMES_DISTRACTED: Final[int] = 45
"""Dikkat dağınıklığı onayı için gereken ardışık kare sayısı (yaklaşık 1.5 sn)."""


# =============================================================================
#  Sonuç Veri Sınıfı
# =============================================================================

@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """``DecisionEngine.evaluate()`` tarafından döndürülen sonuç.

    Attributes:
        status: Sürücünün anlık durumu.
        label: Kullanıcıya gösterilecek Türkçe durum metni.
        color: Durum renk kodu (Hex formatı).
        fatigue_score: 0-100 arası normalize yorgunluk skoru.
        ear: Anlık EAR değeri.
        mar: Anlık MAR değeri.
        yaw: Anlık yaw açısı (derece).
        pitch: Anlık pitch açısı (derece).
    """
    status: DriverStatus
    label: str
    color: str
    fatigue_score: int
    ear: float
    mar: float
    yaw: float
    pitch: float


# =============================================================================
#  Ardışık Kare Sayacı
# =============================================================================

@dataclass
class _FrameCounter:
    """Belirli bir koşulun ardışık kare boyunca sağlanıp sağlanmadığını takip eder.

    Attributes:
        count: Mevcut ardışık kare sayısı.
        threshold: Eşik olarak kabul edilecek ardışık kare sayısı.
        triggered: Eşik aşıldığında ``True`` olur.
    """
    threshold: int
    count: int = field(default=0, init=False)
    triggered: bool = field(default=False, init=False)

    def update(self, condition_met: bool) -> bool:
        """Sayacı günceller.

        Args:
            condition_met: Bu karede koşulun sağlanıp sağlanmadığı.

        Returns:
            Sayaç eşiğe ulaştıysa ``True``, aksi halde ``False``.
        """
        if condition_met:
            self.count += 1
        else:
            self.count = 0
            self.triggered = False

        if self.count >= self.threshold:
            self.triggered = True

        return self.triggered

    def reset(self) -> None:
        """Sayacı sıfırlar."""
        self.count = 0
        self.triggered = False


# =============================================================================
#  Karar Motoru
# =============================================================================

class DecisionEngine:
    """Sürücü durumunu belirleyen merkezi karar motoru.

    Gelen anlık metrik değerlerini (EAR, MAR, Yaw, Pitch) yapılandırma
    eşikleriyle karşılaştırır. Yanlış alarmları önlemek için ardışık kare
    sayacı (oylama filtresi) kullanır.

    Attributes:
        _alert_service: Sesli uyarı tetiklemek için kullanılan servis.
            ``None`` ise sesli uyarılar devre dışıdır.
        _drowsy_counter: Uyku tespiti kare sayacı.
        _yawn_counter: Esneme tespiti kare sayacı.
        _distracted_counter: Dikkat dağınıklığı kare sayacı.
        _ear_threshold: EAR eşik değeri.
        _mar_threshold: MAR eşik değeri.
        _yaw_threshold: Yaw açısı eşik değeri (derece).
        _pitch_threshold: Pitch açısı eşik değeri (derece).
    """

    def __init__(
        self,
        alert_service: object | None = None,
        ear_threshold: float = config.EAR_THRESHOLD,
        mar_threshold: float = config.MAR_THRESHOLD,
        yaw_threshold: float = config.HEAD_YAW_THRESHOLD,
        pitch_threshold: float = config.HEAD_PITCH_THRESHOLD,
        consecutive_drowsy: int = config.CONSECUTIVE_FRAMES_DROWSY,
        consecutive_yawn: int = config.CONSECUTIVE_FRAMES_YAWN,
        consecutive_distracted: int = _CONSECUTIVE_FRAMES_DISTRACTED,
    ) -> None:
        """DecisionEngine nesnesini başlatır.

        Args:
            alert_service: ``AlertService`` nesnesi. ``None`` ise sesli
                uyarılar tetiklenmez.
            ear_threshold: EAR eşik değeri.
            mar_threshold: MAR eşik değeri.
            yaw_threshold: Yaw açısı eşik değeri (derece).
            pitch_threshold: Pitch açısı eşik değeri (derece).
            consecutive_drowsy: Uyku onayı için ardışık kare sayısı.
            consecutive_yawn: Esneme onayı için ardışık kare sayısı.
            consecutive_distracted: Dikkat dağınıklığı onayı için ardışık
                kare sayısı.
        """
        self._alert_service = alert_service

        # Eşik değerleri
        self._ear_threshold: float = ear_threshold
        self._mar_threshold: float = mar_threshold
        self._yaw_threshold: float = yaw_threshold
        self._pitch_threshold: float = pitch_threshold

        # Ardışık kare sayaçları
        self._drowsy_counter = _FrameCounter(threshold=consecutive_drowsy)
        self._yawn_counter = _FrameCounter(threshold=consecutive_yawn)
        self._distracted_counter = _FrameCounter(threshold=consecutive_distracted)

    # ─────────────────────────────────────────────────────────────────────
    #  Ana Değerlendirme
    # ─────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        ear: float,
        mar: float,
        yaw: float,
        pitch: float,
        is_phone_used: bool = False,
        gaze_state: str = "MERKEZ",
    ) -> EvaluationResult:
        """Anlık metrikleri değerlendirerek sürücü durumunu belirler.

        Öncelik sıralaması (en yüksekten düşüğe):
            1. DROWSY  (göz kapanıklığı)
            2. YAWNING (esneme)
            3. DISTRACTED (dikkat dağınıklığı)
            4. NORMAL

        Args:
            ear: Anlık Eye Aspect Ratio değeri (her iki göz ortalaması).
            mar: Anlık Mouth Aspect Ratio değeri.
            yaw: Anlık kafa yaw açısı (derece).
            pitch: Anlık kafa pitch açısı (derece).
            is_phone_used: Telefon kullanım durumu.
            gaze_state: Bakış yönü ("MERKEZ", "SOL", "SAĞ").

        Returns:
            ``EvaluationResult`` nesnesi: durum, etiket, renk, yorgunluk
            skoru ve orijinal metrik değerlerini içerir.
        """        # ── Koşul kontrolleri ──
        is_eyes_closed = ear < self._ear_threshold
        is_yawning = mar > self._mar_threshold
        is_distracted = (
            abs(yaw) > self._yaw_threshold
            or pitch < -self._pitch_threshold
            or gaze_state in ("SOL", "SAĞ")
        )
        # ── Sayaçları güncelle ──
        drowsy_triggered = self._drowsy_counter.update(is_eyes_closed)
        yawn_triggered = self._yawn_counter.update(is_yawning)
        distracted_triggered = self._distracted_counter.update(is_distracted)

        # ── Durum belirleme (öncelik sırasıyla) ──
        status = DriverStatus.NORMAL

        if is_phone_used:
            status = DriverStatus.PHONE_USAGE
        elif drowsy_triggered:
            status = DriverStatus.DROWSY
        elif yawn_triggered:
            status = DriverStatus.YAWNING
        elif distracted_triggered:
            status = DriverStatus.DISTRACTED

        # ── Uyarı tetikleme ──
        if status != DriverStatus.NORMAL:
            self._fire_alert(status)

        # ── Yorgunluk skoru hesaplama ──
        fatigue_score = self._compute_fatigue_score(
            ear, mar, yaw, pitch,
            drowsy_triggered, yawn_triggered, distracted_triggered,
        )

        return EvaluationResult(
            status=status,
            label=_STATUS_LABELS[status],
            color=_STATUS_COLORS[status],
            fatigue_score=fatigue_score,
            ear=ear,
            mar=mar,
            yaw=yaw,
            pitch=pitch,
        )

    # ─────────────────────────────────────────────────────────────────────
    #  Yorgunluk Skoru
    # ─────────────────────────────────────────────────────────────────────

    def _compute_fatigue_score(
        self,
        ear: float,
        mar: float,
        yaw: float,
        pitch: float,
        drowsy: bool,
        yawning: bool,
        distracted: bool,
    ) -> int:
        """0-100 arası normalize edilmiş yorgunluk skoru hesaplar.

        Bileşenler:
            - EAR katkısı (ağırlık: %40) : Eşik altına düştükçe artar.
            - MAR katkısı (ağırlık: %25) : Eşik üstüne çıktıkça artar.
            - Head pose katkısı (ağırlık: %15) : Sapma büyüdükçe artar.
            - Tetikleme bonusu (ağırlık: %20) : Onaylanmış durumlar ekler.

        Returns:
            0 (tamamen uyanık) ile 100 (yüksek yorgunluk) arasında tam sayı.
        """
        # EAR katkısı: eşiğin ne kadar altındaysa o kadar yüksek
        if self._ear_threshold > 0:
            ear_ratio = max(0.0, 1.0 - (ear / self._ear_threshold))
        else:
            ear_ratio = 0.0
        ear_score = ear_ratio * 40.0

        # MAR katkısı: eşiğin ne kadar üstündeyse o kadar yüksek
        if self._mar_threshold > 0:
            mar_ratio = max(0.0, (mar / self._mar_threshold) - 1.0)
        else:
            mar_ratio = 0.0
        mar_score = min(mar_ratio, 1.0) * 25.0

        # Head pose katkısı
        yaw_ratio = min(abs(yaw) / max(self._yaw_threshold, 1.0), 1.0)
        pitch_neg = max(-pitch, 0.0)
        pitch_ratio = min(pitch_neg / max(self._pitch_threshold, 1.0), 1.0)
        head_score = max(yaw_ratio, pitch_ratio) * 15.0

        # Tetikleme bonusu
        bonus = 0.0
        if drowsy:
            bonus = 20.0
        elif yawning:
            bonus = 12.0
        elif distracted:
            bonus = 8.0

        total = ear_score + mar_score + head_score + bonus
        return max(0, min(100, int(round(total))))

    # ─────────────────────────────────────────────────────────────────────
    #  Uyarı Tetikleme
    # ─────────────────────────────────────────────────────────────────────

    def _fire_alert(self, status: DriverStatus) -> None:
        """İlgili durum için AlertService üzerinden uyarı tetikler.

        Args:
            status: Tetiklenecek uyarı durumu.
        """
        if self._alert_service is None:
            return

        category = _STATUS_ALERT_CATEGORY.get(status)
        if category is None:
            return

        try:
            self._alert_service.trigger(category)
        except Exception:
            logger.exception(
                "AlertService tetikleme hatası (kategori: %s)", category,
            )

    # ─────────────────────────────────────────────────────────────────────
    #  Yardımcı
    # ─────────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Tüm ardışık kare sayaçlarını sıfırlar.

        Yüz kaybolduğunda veya yeni bir oturum başladığında çağrılmalıdır.
        """
        self._drowsy_counter.reset()
        self._yawn_counter.reset()
        self._distracted_counter.reset()
        logger.debug("DecisionEngine sayaçları sıfırlandı.")
