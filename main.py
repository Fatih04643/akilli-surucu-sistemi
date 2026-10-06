"""
Akıllı Sürücü Yorgunluk ve Dikkat Dağınıklığı Tespit Sistemi – Giriş Noktası.

QApplication başlatır, ana HUD penceresini oluşturur ve olay döngüsünü çalıştırır.

Kullanım:
    python main.py
"""

from __future__ import annotations

import os
import sys

# PyQt5 yüklenmeden önce PyTorch'un dll'lerini hafızaya alıp çakışmayı önleyelim
import torch
from ultralytics import YOLO

torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
if os.path.exists(torch_lib):
    os.add_dll_directory(torch_lib)

import logging

# ── Qt Platform Plugin Yolu ──────────────────────────────────────────────────
# PyQt5'in platform eklentilerini (qwindows.dll) bulabilmesi için
# QT_QPA_PLATFORM_PLUGIN_PATH ortam değişkenini ayarla.
# Bu blok, tüm PyQt5 importlarından ÖNCE çalışmalıdır.
def _setup_qt_plugin_path() -> None:
    """PyQt5 platform eklenti dizinini ortam değişkenine atar."""
    try:
        import PyQt5
        qt_dir = os.path.dirname(PyQt5.__file__)
        # PyQt5 >= 5.15.4 → Qt5/plugins/platforms
        candidates = [
            os.path.join(qt_dir, "Qt5", "plugins", "platforms"),
            os.path.join(qt_dir, "Qt", "plugins", "platforms"),
            os.path.join(qt_dir, "plugins", "platforms"),
        ]
        for path in candidates:
            if os.path.isdir(path):
                os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = path
                return
    except ImportError:
        pass

_setup_qt_plugin_path()
# ─────────────────────────────────────────────────────────────────────────────

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from ui.main_window import MainWindow


def _configure_logging() -> None:
    """Uygulama genelinde loglama ayarlarını yapılandırır."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    """Uygulamayı başlatır ve olay döngüsünü çalıştırır.

    Returns:
        Uygulama çıkış kodu.
    """
    _configure_logging()
    logger = logging.getLogger(__name__)
    logger.info("Akıllı Sürücü Sistemi başlatılıyor…")

    # Yüksek DPI ekranlar için ölçekleme
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)

    # Uygulama genelinde varsayılan font
    font = QFont("Segoe UI", 10)
    font.setHintingPreference(QFont.PreferFullHinting)
    app.setFont(font)

    # Ana pencereyi oluştur ve göster
    window = MainWindow()
    window.show()

    logger.info("Arayüz hazır – olay döngüsü başlıyor.")
    exit_code = app.exec_()

    logger.info("Uygulama kapatıldı (çıkış kodu: %d).", exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
