"""
Pipeline Birim Testleri.

``calculate_ear``, ``calculate_mar`` ve ``DecisionEngine`` bileşenlerinin
doğru çalıştığını doğrulayan kapsamlı pytest test seti.

Çalıştırma:
    $ cd Akıllı_Sürücü_Sistemi
    $ python -m pytest tests/test_pipeline.py -v
"""

from __future__ import annotations

import math

import pytest

from core.metrics import calculate_ear, calculate_mar
from core.decision_engine import DecisionEngine, DriverStatus, EvaluationResult


# =============================================================================
#  Sentetik Landmark Veri Oluşturucular
# =============================================================================

def _make_open_eye_landmarks() -> list[tuple[float, float]]:
    """Göz açık durumu için sentetik 6-nokta landmark üretir.

    Beklenen EAR: yüksek (≈ 1.0)

    Düzen: [P1, P2, P3, P4, P5, P6]
        P1(0,0) ---- P4(4,0)   (yatay eksen)
        P2(1,2), P3(3,2)       (üst kapak)
        P5(3,-2), P6(1,-2)     (alt kapak)
    """
    return [
        (0.0, 0.0),   # P1 - sol köşe
        (1.0, 2.0),   # P2 - üst kapak (dış)
        (3.0, 2.0),   # P3 - üst kapak (iç)
        (4.0, 0.0),   # P4 - sağ köşe
        (3.0, -2.0),  # P5 - alt kapak (iç)
        (1.0, -2.0),  # P6 - alt kapak (dış)
    ]


def _make_closed_eye_landmarks() -> list[tuple[float, float]]:
    """Göz kapalı durumu için sentetik 6-nokta landmark üretir.

    Beklenen EAR: çok düşük (≈ 0.025)

    Dikey mesafeler sıfıra yakın tutularak kapalı göz simüle edilir.
    """
    return [
        (0.0, 0.0),    # P1
        (1.0, 0.05),   # P2 - neredeyse kapanmış üst
        (3.0, 0.05),   # P3
        (4.0, 0.0),    # P4
        (3.0, -0.05),  # P5 - neredeyse kapanmış alt
        (1.0, -0.05),  # P6
    ]


def _make_mouth_closed_landmarks() -> list[tuple[float, float]]:
    """Ağız kapalı durumu için sentetik landmark üretir.

    Beklenen MAR: düşük (< 0.65)
    """
    return [
        (0.0, 0.0),    # P1 - sol köşe
        (1.0, 0.2),    # P2 - üst dudak
        (2.0, 0.2),    # P3 - sağ üst
        (3.0, 0.0),    # P4 - sağ köşe
        (2.0, -0.2),   # P5 - alt dudak
        (1.0, -0.2),   # P6 - üst dudak alt kenarı
    ]


def _make_mouth_yawning_landmarks() -> list[tuple[float, float]]:
    """Esneme (ağız açık) durumu için sentetik landmark üretir.

    Beklenen MAR: yüksek (> 0.65)
    """
    return [
        (0.0, 0.0),    # P1 - sol köşe
        (1.0, 3.0),    # P2 - üst dudak (çok açık)
        (2.0, 3.0),    # P3
        (3.0, 0.0),    # P4 - sağ köşe
        (2.0, -3.0),   # P5 - alt dudak (çok açık)
        (1.0, -3.0),   # P6
    ]


# =============================================================================
#  Test: calculate_ear
# =============================================================================

class TestCalculateEAR:
    """``calculate_ear`` fonksiyonunun birim testleri."""

    def test_open_eye_returns_high_ear(self) -> None:
        """Göz açıkken EAR değeri yüksek (> 0.3) olmalıdır."""
        landmarks = _make_open_eye_landmarks()
        ear = calculate_ear(landmarks)
        assert ear > 0.3, f"Açık göz için EAR çok düşük: {ear:.4f}"

    def test_closed_eye_returns_low_ear(self) -> None:
        """Göz kapalıyken EAR değeri düşük (< 0.1) olmalıdır."""
        landmarks = _make_closed_eye_landmarks()
        ear = calculate_ear(landmarks)
        assert ear < 0.1, f"Kapalı göz için EAR çok yüksek: {ear:.4f}"

    def test_open_ear_greater_than_closed(self) -> None:
        """Açık göz EAR'ı her zaman kapalı göz EAR'ından büyük olmalıdır."""
        ear_open = calculate_ear(_make_open_eye_landmarks())
        ear_closed = calculate_ear(_make_closed_eye_landmarks())
        assert ear_open > ear_closed

    def test_invalid_landmark_count_raises(self) -> None:
        """Yanlış sayıda landmark verildiğinde ValueError fırlatılmalıdır."""
        with pytest.raises(ValueError, match="6 landmark gerekli"):
            calculate_ear([(0, 0), (1, 1)])

    def test_zero_horizontal_distance_returns_zero(self) -> None:
        """P1 ve P4 aynı noktadaysa (yatay mesafe ≈ 0) EAR 0.0 olmalıdır."""
        landmarks = [
            (0.0, 0.0),  # P1
            (0.0, 1.0),  # P2
            (0.0, 1.0),  # P3
            (0.0, 0.0),  # P4 = P1
            (0.0, -1.0), # P5
            (0.0, -1.0), # P6
        ]
        assert calculate_ear(landmarks) == 0.0


# =============================================================================
#  Test: calculate_mar
# =============================================================================

class TestCalculateMAR:
    """``calculate_mar`` fonksiyonunun birim testleri."""

    def test_closed_mouth_returns_low_mar(self) -> None:
        """Ağız kapalıyken MAR değeri düşük (< 0.3) olmalıdır."""
        landmarks = _make_mouth_closed_landmarks()
        mar = calculate_mar(landmarks)
        assert mar < 0.3, f"Kapalı ağız için MAR çok yüksek: {mar:.4f}"

    def test_yawning_mouth_returns_high_mar(self) -> None:
        """Esneme sırasında MAR değeri yüksek (> 0.65) olmalıdır."""
        landmarks = _make_mouth_yawning_landmarks()
        mar = calculate_mar(landmarks)
        assert mar > 0.65, f"Esneme için MAR çok düşük: {mar:.4f}"

    def test_yawning_mar_greater_than_closed(self) -> None:
        """Esneme MAR'ı her zaman kapalı ağız MAR'ından büyük olmalıdır."""
        mar_yawning = calculate_mar(_make_mouth_yawning_landmarks())
        mar_closed = calculate_mar(_make_mouth_closed_landmarks())
        assert mar_yawning > mar_closed

    def test_invalid_landmark_count_raises(self) -> None:
        """Yanlış sayıda landmark verildiğinde ValueError fırlatılmalıdır."""
        with pytest.raises(ValueError, match="6 landmark gerekli"):
            calculate_mar([(0, 0)])


# =============================================================================
#  Test: DecisionEngine
# =============================================================================

class TestDecisionEngine:
    """``DecisionEngine`` sınıfının durum geçiş ve oylama testleri."""

    @staticmethod
    def _normal_metrics() -> dict[str, float]:
        """Tamamen normal sürücü metrikleri."""
        return {"ear": 0.32, "mar": 0.30, "yaw": 5.0, "pitch": 0.0}

    @staticmethod
    def _drowsy_metrics() -> dict[str, float]:
        """Göz kapalı (uykulu) sürücü metrikleri."""
        return {"ear": 0.15, "mar": 0.30, "yaw": 5.0, "pitch": 0.0}

    @staticmethod
    def _yawning_metrics() -> dict[str, float]:
        """Esneme sürücü metrikleri."""
        return {"ear": 0.32, "mar": 0.80, "yaw": 5.0, "pitch": 0.0}

    @staticmethod
    def _distracted_yaw_metrics() -> dict[str, float]:
        """Kafa yana dönmüş (dikkat dağınık) sürücü metrikleri."""
        return {"ear": 0.32, "mar": 0.30, "yaw": 45.0, "pitch": 0.0}

    @staticmethod
    def _distracted_pitch_metrics() -> dict[str, float]:
        """Kafa aşağı eğik (telefona bakıyor) sürücü metrikleri."""
        return {"ear": 0.32, "mar": 0.30, "yaw": 5.0, "pitch": -25.0}

    def test_normal_state_when_all_metrics_ok(self) -> None:
        """Tüm metrikler normal aralıktayken durum NORMAL olmalıdır."""
        engine = DecisionEngine(alert_service=None)

        for _ in range(30):
            result = engine.evaluate(**self._normal_metrics())

        assert result.status == DriverStatus.NORMAL
        assert result.color == "#00C853"

    def test_drowsy_after_consecutive_frames(self) -> None:
        """EAR eşik altında ardışık kare sayısı kadar kalırsa DROWSY olmalıdır."""
        consecutive = 5
        engine = DecisionEngine(
            alert_service=None,
            consecutive_drowsy=consecutive,
        )

        # İlk (consecutive - 1) karede henüz DROWSY olmamalı
        for i in range(consecutive - 1):
            result = engine.evaluate(**self._drowsy_metrics())
            assert result.status == DriverStatus.NORMAL, (
                f"Kare {i+1}/{consecutive}: Henüz DROWSY olmamalıydı"
            )

        # Tam eşik karesinde DROWSY olmalı
        result = engine.evaluate(**self._drowsy_metrics())
        assert result.status == DriverStatus.DROWSY

    def test_drowsy_resets_on_normal_frame(self) -> None:
        """Aradaki normal bir kare sayacı sıfırlamalıdır."""
        consecutive = 5
        engine = DecisionEngine(
            alert_service=None,
            consecutive_drowsy=consecutive,
        )

        # 3 kare uykulu
        for _ in range(3):
            engine.evaluate(**self._drowsy_metrics())

        # 1 kare normal → sayaç sıfırlanır
        engine.evaluate(**self._normal_metrics())

        # Tekrar 4 kare uykulu → hâlâ eşiğe ulaşmadı
        for _ in range(4):
            result = engine.evaluate(**self._drowsy_metrics())

        assert result.status == DriverStatus.NORMAL

    def test_yawning_after_consecutive_frames(self) -> None:
        """MAR eşik üstünde ardışık kare sayısı kadar kalırsa YAWNING olmalıdır."""
        consecutive = 5
        engine = DecisionEngine(
            alert_service=None,
            consecutive_yawn=consecutive,
        )

        for _ in range(consecutive - 1):
            result = engine.evaluate(**self._yawning_metrics())
            assert result.status == DriverStatus.NORMAL

        result = engine.evaluate(**self._yawning_metrics())
        assert result.status == DriverStatus.YAWNING

    def test_distracted_yaw_after_consecutive_frames(self) -> None:
        """Yaw eşik üstünde ardışık 15 kare sürerse DISTRACTED olmalıdır."""
        consecutive = 5
        engine = DecisionEngine(
            alert_service=None,
            consecutive_distracted=consecutive,
        )

        for _ in range(consecutive - 1):
            result = engine.evaluate(**self._distracted_yaw_metrics())
            assert result.status == DriverStatus.NORMAL

        result = engine.evaluate(**self._distracted_yaw_metrics())
        assert result.status == DriverStatus.DISTRACTED

    def test_distracted_pitch_after_consecutive_frames(self) -> None:
        """Negatif pitch eşik üstünde ardışık kare sürerse DISTRACTED olmalıdır."""
        consecutive = 5
        engine = DecisionEngine(
            alert_service=None,
            consecutive_distracted=consecutive,
        )

        for _ in range(consecutive):
            result = engine.evaluate(**self._distracted_pitch_metrics())

        assert result.status == DriverStatus.DISTRACTED

    def test_drowsy_has_priority_over_yawning(self) -> None:
        """DROWSY durumu YAWNING'den öncelikli olmalıdır."""
        consecutive = 3
        engine = DecisionEngine(
            alert_service=None,
            consecutive_drowsy=consecutive,
            consecutive_yawn=consecutive,
        )

        # Hem göz kapalı hem esneme metrikleri
        mixed_metrics = {"ear": 0.15, "mar": 0.80, "yaw": 5.0, "pitch": 0.0}

        for _ in range(consecutive):
            result = engine.evaluate(**mixed_metrics)

        assert result.status == DriverStatus.DROWSY

    def test_fatigue_score_in_valid_range(self) -> None:
        """Yorgunluk skoru her zaman 0-100 arasında olmalıdır."""
        engine = DecisionEngine(alert_service=None)

        test_cases = [
            self._normal_metrics(),
            self._drowsy_metrics(),
            self._yawning_metrics(),
            self._distracted_yaw_metrics(),
            {"ear": 0.0, "mar": 2.0, "yaw": 90.0, "pitch": -45.0},
        ]

        for metrics in test_cases:
            result = engine.evaluate(**metrics)
            assert 0 <= result.fatigue_score <= 100, (
                f"Skor aralık dışı: {result.fatigue_score} (metrikler: {metrics})"
            )

    def test_evaluation_result_contains_metrics(self) -> None:
        """EvaluationResult döndürülen metrikleri doğru taşımalıdır."""
        engine = DecisionEngine(alert_service=None)
        metrics = self._normal_metrics()
        result = engine.evaluate(**metrics)

        assert result.ear == pytest.approx(metrics["ear"])
        assert result.mar == pytest.approx(metrics["mar"])
        assert result.yaw == pytest.approx(metrics["yaw"])
        assert result.pitch == pytest.approx(metrics["pitch"])

    def test_reset_clears_counters(self) -> None:
        """``reset()`` çağrıldığında tüm sayaçlar sıfırlanmalıdır."""
        consecutive = 5
        engine = DecisionEngine(
            alert_service=None,
            consecutive_drowsy=consecutive,
        )

        # 4 kare uykulu (eşiğe 1 kare kaldı)
        for _ in range(consecutive - 1):
            engine.evaluate(**self._drowsy_metrics())

        # Reset
        engine.reset()

        # 1 kare daha uykulu → sıfırlandığı için NORMAL olmalı
        result = engine.evaluate(**self._drowsy_metrics())
        assert result.status == DriverStatus.NORMAL
