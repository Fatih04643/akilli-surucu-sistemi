"""
Ana Pencere Modülü – HUD Konseptli Koyu Temalı Dashboard.

Otomotiv HUD (Head-Up Display) estetiğinde tasarlanmış PyQt5 ana penceresi.
Sol/orta panelde canlı kamera akışı, sağ panelde telemetri göstergeleri,
alt panelde ise dinamik EAR/MAR zaman serisi grafiği yer alır.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Final

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon, QColor
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QProgressBar,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QSizePolicy,
    QGraphicsDropShadowEffect,
    QMessageBox,
)
from core.decision_engine import DecisionEngine
from services.alert_service import AlertService
from services.report_service import ReportService
from ui.video_thread import VideoThread
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Sabitler
# ─────────────────────────────────────────────────────────────────────────────

_GRAPH_HISTORY: Final[int] = 100
"""Grafik için tutulan son kare sayısı."""

# ─────────────────────────────────────────────────────────────────────────────
#  Koyu Tema Stil Sayfası (HUD / Otomotiv Konsept)
# ─────────────────────────────────────────────────────────────────────────────

_DARK_THEME_QSS: Final[str] = """
/* ── Genel Pencere ── */
QMainWindow {
    background-color: #0D0D0D;
}
QWidget#centralWidget {
    background-color: #0D0D0D;
}

/* ── Paneller ── */
QFrame#videoPanel {
    background-color: #111111;
    border: 1px solid #1E1E1E;
    border-radius: 12px;
}
QFrame#telemetryPanel {
    background-color: #111111;
    border: 1px solid #1E1E1E;
    border-radius: 12px;
}
QFrame#graphPanel {
    background-color: #111111;
    border: 1px solid #1E1E1E;
    border-radius: 12px;
}

/* ── Durum Kartı ── */
QFrame#statusCard {
    border-radius: 10px;
    padding: 16px;
}

/* ── Metrik Etiketleri ── */
QLabel#metricTitle {
    color: #6B7280;
    font-size: 11px;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 1px;
}
QLabel#metricValue {
    color: #E5E7EB;
    font-size: 22px;
    font-weight: bold;
}

/* ── Yorgunluk Çubuğu ── */
QProgressBar {
    background-color: #1A1A2E;
    border: none;
    border-radius: 6px;
    text-align: center;
    color: #FFFFFF;
    font-weight: bold;
    font-size: 12px;
    min-height: 20px;
}
QProgressBar::chunk {
    border-radius: 6px;
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 #00C853, stop:0.5 #FFD600, stop:1.0 #D50000
    );
}

/* ── Butonlar ── */
QPushButton#btnStart {
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:1,
        stop:0 #00C853, stop:1 #00E676
    );
    color: #0D0D0D;
    border: none;
    border-radius: 8px;
    padding: 12px 24px;
    font-size: 14px;
    font-weight: bold;
    min-height: 36px;
}
QPushButton#btnStart:hover {
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:1,
        stop:0 #00E676, stop:1 #69F0AE
    );
}
QPushButton#btnStart:pressed {
    background: #00C853;
}
QPushButton#btnStart:disabled {
    background: #2E7D32;
    color: #6B7280;
}

QPushButton#btnStop {
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:1,
        stop:0 #D50000, stop:1 #FF1744
    );
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    padding: 12px 24px;
    font-size: 14px;
    font-weight: bold;
    min-height: 36px;
}
QPushButton#btnStop:hover {
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:1,
        stop:0 #FF1744, stop:1 #FF5252
    );
}
QPushButton#btnStop:pressed {
    background: #D50000;
}
QPushButton#btnStop:disabled {
    background: #4E342E;
    color: #6B7280;
}

/* ── Video Etiketi ── */
QLabel#videoLabel {
    color: #4B5563;
    font-size: 16px;
    border: 2px dashed #1E1E1E;
    border-radius: 8px;
}

/* ── Panel Başlıkları ── */
QLabel#panelTitle {
    color: #9CA3AF;
    font-size: 12px;
    font-weight: bold;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* ── Ayırıcı Çizgi ── */
QFrame#separator {
    background-color: #1E1E1E;
    max-height: 1px;
}
"""


class MainWindow(QMainWindow):
    """Akıllı Sürücü Sistemi – HUD Ana Penceresi.

    Koyu temalı, otomotiv konseptli kullanıcı arayüzü. Canlı kamera akışı,
    sürücü durum kartı, metrik göstergeleri, yorgunluk skoru ve zaman serisi
    grafiği içerir.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.setWindowTitle("🚗 Akıllı Sürücü İzleme Sistemi")
        self.setMinimumSize(1280, 780)
        self.resize(1440, 860)
        self.setStyleSheet(_DARK_THEME_QSS)

        # Servisler
        self._alert_service = AlertService()
        self._decision_engine = DecisionEngine(alert_service=self._alert_service)
        self._report_service = ReportService()
        self._video_thread: VideoThread | None = None
        # Grafik veri tamponları
        self._ear_history: deque[float] = deque(maxlen=_GRAPH_HISTORY)
        self._mar_history: deque[float] = deque(maxlen=_GRAPH_HISTORY)

        # UI oluşturma
        self._build_ui()

    # ─────────────────────────────────────────────────────────────────────
    #  UI Oluşturma
    # ─────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        """Tüm arayüz bileşenlerini oluşturur ve düzenler."""
        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(12)

        # ── Üst Bölüm: Video + Telemetri ──
        top_layout = QHBoxLayout()
        top_layout.setSpacing(12)

        top_layout.addWidget(self._build_video_panel(), stretch=3)
        top_layout.addWidget(self._build_telemetry_panel(), stretch=1)

        root_layout.addLayout(top_layout, stretch=3)

        # ── Alt Bölüm: Grafik ──
        root_layout.addWidget(self._build_graph_panel(), stretch=1)

    # ── Video Paneli ──────────────────────────────────────────────────────

    def _build_video_panel(self) -> QFrame:
        """Canlı kamera görüntüsü panelini oluşturur."""
        panel = QFrame()
        panel.setObjectName("videoPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Başlık
        title = QLabel("◉  CANLI KAMERA AKIŞI")
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        # Video etiketi
        self._video_label = QLabel("Kamerayı başlatmak için ▶ BAŞLAT butonuna basın")
        self._video_label.setObjectName("videoLabel")
        self._video_label.setAlignment(Qt.AlignCenter)
        self._video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._video_label.setMinimumSize(400, 300)
        layout.addWidget(self._video_label)

        return panel

    # ── Telemetri Paneli ──────────────────────────────────────────────────

    def _build_telemetry_panel(self) -> QFrame:
        """Sağ taraftaki gösterge ve kontrol panelini oluşturur."""
        panel = QFrame()
        panel.setObjectName("telemetryPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Panel başlığı
        title = QLabel("◈  TELEMETRİ")
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        layout.addWidget(self._make_separator())

        # ── Durum Kartı ──
        self._status_card = QFrame()
        self._status_card.setObjectName("statusCard")
        self._status_card.setStyleSheet(
            "background-color: #1B5E20; border-radius: 10px; padding: 16px;"
        )
        status_layout = QVBoxLayout(self._status_card)
        status_layout.setContentsMargins(12, 12, 12, 12)
        status_layout.setSpacing(4)

        status_label_title = QLabel("SÜRÜCÜ DURUMU")
        status_label_title.setStyleSheet(
            "color: rgba(255,255,255,0.6); font-size: 10px; font-weight: bold; "
            "letter-spacing: 2px;"
        )
        status_label_title.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(status_label_title)

        self._status_text = QLabel("HAZIR")
        self._status_text.setAlignment(Qt.AlignCenter)
        self._status_text.setStyleSheet(
            "color: #FFFFFF; font-size: 18px; font-weight: bold;"
        )
        self._status_text.setWordWrap(True)
        status_layout.addWidget(self._status_text)

        # Glow efekti
        glow = QGraphicsDropShadowEffect()
        glow.setBlurRadius(30)
        glow.setOffset(0, 0)
        glow.setColor(QColor(0, 200, 83, 120))
        self._status_card.setGraphicsEffect(glow)
        self._status_glow = glow

        layout.addWidget(self._status_card)

        layout.addWidget(self._make_separator())

        # ── Yorgunluk Skoru ──
        fatigue_title = QLabel("YORGUNLUK SKORU")
        fatigue_title.setObjectName("metricTitle")
        layout.addWidget(fatigue_title)

        self._fatigue_bar = QProgressBar()
        self._fatigue_bar.setRange(0, 100)
        self._fatigue_bar.setValue(0)
        self._fatigue_bar.setFormat("%v / 100")
        layout.addWidget(self._fatigue_bar)

        layout.addWidget(self._make_separator())

        # ── Metrik Göstergeleri ──
        metrics_title = QLabel("ANLIK METRİKLER")
        metrics_title.setObjectName("metricTitle")
        layout.addWidget(metrics_title)

        self._ear_label = self._make_metric_row("EAR", "0.00")
        self._mar_label = self._make_metric_row("MAR", "0.00")
        self._yaw_label = self._make_metric_row("YAW", "0.0°")
        self._pitch_label = self._make_metric_row("PITCH", "0.0°")
        self._gaze_label = self._make_metric_row("BAKIŞ", "MERKEZ")

        for row_widget, _ in (
            self._ear_label, self._mar_label,
            self._yaw_label, self._pitch_label,
            self._gaze_label,
        ):
            layout.addWidget(row_widget)
        layout.addStretch()

        layout.addWidget(self._make_separator())

        # ── Kontrol Butonları ──
        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(8)

        self._btn_start = QPushButton("▶  BAŞLAT")
        self._btn_start.setObjectName("btnStart")
        self._btn_start.setCursor(Qt.PointingHandCursor)
        self._btn_start.clicked.connect(self._on_start)
        btn_layout.addWidget(self._btn_start)

        self._btn_stop = QPushButton("■  DURDUR")
        self._btn_stop.setObjectName("btnStop")
        self._btn_stop.setCursor(Qt.PointingHandCursor)
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop)
        btn_layout.addWidget(self._btn_stop)
        
        self._btn_report = QPushButton("📊  SÜRÜŞÜ TAMAMLA & RAPOR AL")
        self._btn_report.setObjectName("btnStart") # Start butonu stiliyle aynı olsun
        self._btn_report.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1976D2, stop:1 #42A5F5); color: white;")
        self._btn_report.setCursor(Qt.PointingHandCursor)
        self._btn_report.setEnabled(False)
        self._btn_report.clicked.connect(self._on_report)
        btn_layout.addWidget(self._btn_report)
        layout.addLayout(btn_layout)

        return panel

    # ── Grafik Paneli ─────────────────────────────────────────────────────

    def _build_graph_panel(self) -> QFrame:
        """Alt kısımdaki EAR/MAR zaman serisi grafiğini oluşturur."""
        panel = QFrame()
        panel.setObjectName("graphPanel")
        panel.setMinimumHeight(200)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        title = QLabel("◆  ZAMAN SERİSİ GRAFİĞİ  (Son 100 Kare)")
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        # pyqtgraph ayarları
        pg.setConfigOptions(antialias=True)

        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground("#0D0D0D")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.15)
        self._plot_widget.setLabel("left", "Değer", color="#9CA3AF")
        self._plot_widget.setLabel("bottom", "Kare #", color="#9CA3AF")
        self._plot_widget.setYRange(0, 1.0, padding=0.05)
        self._plot_widget.getPlotItem().layout.setContentsMargins(10, 10, 20, 10)
        
        self._plot_widget.addLegend(
            offset=(20, 20),
            labelTextColor="#E5E7EB",
            brush=pg.mkBrush(17, 17, 17, 150),
            pen=pg.mkPen(color="#1E1E1E", width=1),
        )

        # EAR eğrisi – camgöbeği / cyan
        self._ear_curve = self._plot_widget.plot(
            pen=pg.mkPen(color=(0, 229, 255), width=2),
            name="EAR",
        )
        # MAR eğrisi – turuncu
        self._mar_curve = self._plot_widget.plot(
            pen=pg.mkPen(color=(255, 167, 38), width=2),
            name="MAR",
        )

        # EAR eşik çizgisi
        ear_threshold_line = pg.InfiniteLine(
            pos=0.25, angle=0,
            pen=pg.mkPen(color=(0, 229, 255, 80), width=1, style=Qt.DashLine),
            label="EAR eşik",
            labelOpts={"color": (0, 229, 255, 120), "position": 0.95},
        )
        self._plot_widget.addItem(ear_threshold_line)

        # MAR eşik çizgisi
        mar_threshold_line = pg.InfiniteLine(
            pos=0.65, angle=0,
            pen=pg.mkPen(color=(255, 167, 38, 80), width=1, style=Qt.DashLine),
            label="MAR eşik",
            labelOpts={"color": (255, 167, 38, 120), "position": 0.05},
        )
        self._plot_widget.addItem(mar_threshold_line)

        layout.addWidget(self._plot_widget)

        return panel

    # ─────────────────────────────────────────────────────────────────────
    #  Yardımcı Widget Oluşturucular
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _make_separator() -> QFrame:
        """Yatay ayırıcı çizgi oluşturur."""
        sep = QFrame()
        sep.setObjectName("separator")
        sep.setFrameShape(QFrame.HLine)
        sep.setFixedHeight(1)
        return sep

    @staticmethod
    def _make_metric_row(
        label_text: str,
        initial_value: str,
    ) -> tuple[QWidget, QLabel]:
        """Metrik başlık + değer satırı oluşturur.

        Returns:
            (satır_widget, değer_label) çifti.
        """
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 2, 0, 2)
        row_layout.setSpacing(8)

        title = QLabel(label_text)
        title.setObjectName("metricTitle")
        title.setFixedWidth(60)
        row_layout.addWidget(title)

        value = QLabel(initial_value)
        value.setObjectName("metricValue")
        value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row_layout.addWidget(value)

        return row, value

    # ─────────────────────────────────────────────────────────────────────
    #  Slot'lar – Buton Eylemleri
    # ─────────────────────────────────────────────────────────────────────

    def _on_start(self) -> None:
        """Kamerayı ve video thread'ini başlatır."""
        if self._video_thread is not None and self._video_thread.isRunning():
            return

        # Alert servisi başlat
        self._alert_service.start()

        # Video thread oluştur ve sinyalleri bağla
        self._video_thread = VideoThread(
            camera_index=0,
            decision_engine=self._decision_engine,
            report_service=self._report_service,
        )
        self._video_thread.frame_ready.connect(self._on_frame_ready)
        self._video_thread.metrics_ready.connect(self._on_metrics_ready)
        self._video_thread.status_ready.connect(self._on_status_ready)
        self._video_thread.gaze_ready.connect(self._on_gaze_ready)
        self._video_thread.finished.connect(self._on_thread_finished)
        self._video_thread.start()

        self._report_service.start_session()

        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._btn_report.setEnabled(True)
        logger.info("Kamera akışı başlatıldı.")
    def _on_stop(self) -> None:
        """Video thread'ini durdurur."""
        if self._video_thread is not None:
            self._video_thread.stop()
            self._video_thread.wait(5000)

        self._alert_service.shutdown()
        self._decision_engine.reset()

        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)

        self._video_label.clear()
        self._video_label.setText("Kamera durduruldu")
        logger.info("Kamera akışı durduruldu.")

    def _on_report(self) -> None:
        """Sürüşü tamamlar ve rapor gösterir."""
        import os
        
        if self._video_thread is not None and self._video_thread.isRunning():
            self._on_stop()
        
        report, pdf_path = self._report_service.end_session()
        self._btn_report.setEnabled(False)
        
        if report and pdf_path:
            # Durum etiketini güncelle
            self._status_text.setText("PDF Raporu Oluşturuldu ve Açıldı")
            
            # PDF'i varsayılan okuyucu ile aç
            try:
                os.startfile(str(pdf_path))
            except Exception as e:
                logger.error(f"PDF açılamadı: {e}")
                self._status_text.setText("PDF Oluşturuldu (Açılamadı)")

    def _on_thread_finished(self) -> None:
        """Video thread sonlandığında çağrılır."""
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)

    # ─────────────────────────────────────────────────────────────────────
    #  Slot'lar – Sinyal Alıcıları
    # ─────────────────────────────────────────────────────────────────────

    def _on_frame_ready(self, image: QImage) -> None:
        """Yeni bir kare geldiğinde video etiketini günceller."""
        pixmap = QPixmap.fromImage(image).scaled(
            self._video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._video_label.setPixmap(pixmap)

    def _on_metrics_ready(
        self,
        ear: float,
        mar: float,
        yaw: float,
        pitch: float,
    ) -> None:
        """Anlık metrik değerlerini günceller ve grafiği yeniler."""
        # Etiketleri güncelle
        _, ear_val = self._ear_label
        _, mar_val = self._mar_label
        _, yaw_val = self._yaw_label
        _, pitch_val = self._pitch_label

        ear_val.setText(f"{ear:.3f}")
        mar_val.setText(f"{mar:.3f}")
        yaw_val.setText(f"{yaw:.1f}°")
        pitch_val.setText(f"{pitch:.1f}°")

        # Grafik verisini güncelle
        self._ear_history.append(ear)
        self._mar_history.append(mar)

        x = np.arange(len(self._ear_history))
        self._ear_curve.setData(x, list(self._ear_history))
        self._mar_curve.setData(x, list(self._mar_history))

    def _on_status_ready(
        self,
        status_text: str,
        hex_color: str,
        fatigue_score: int,
    ) -> None:
        """Sürücü durum kartını ve yorgunluk çubuğunu günceller."""
        self._status_text.setText(status_text)
        self._fatigue_bar.setValue(fatigue_score)

        # Durum kartı arka plan rengini güncelle
        # Hex'ten biraz koyu bir arka plan + orijinal renkte glow oluştur
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)

        # Koyu arka plan (orijinal rengin %20 opaklığı)
        bg_color = f"rgba({r}, {g}, {b}, 40)"
        border_color = f"rgba({r}, {g}, {b}, 120)"
        self._status_card.setStyleSheet(
            f"background-color: {bg_color}; "
            f"border: 1px solid {border_color}; "
            f"border-radius: 10px; padding: 16px;"
        )

        # Glow efekti rengini güncelle
        self._status_glow.setColor(QColor(r, g, b, 140))

    def _on_gaze_ready(self, gaze_state: str) -> None:
        """Bakış yönü etiketini günceller."""
        _, gaze_val = self._gaze_label
        gaze_val.setText(gaze_state)

    # ─────────────────────────────────────────────────────────────────────
    #  Pencere Kapatma
    # ─────────────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        """Pencere kapatılırken tüm kaynakları temizler."""
        self._on_stop()
        event.accept()
