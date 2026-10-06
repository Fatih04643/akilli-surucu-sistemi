"""
Sesli Uyarı Servisi.

UI ve görüntü işleme döngüsünü bloke etmeden arka plan thread'inde
Türkçe sesli ikazlar çalıştıran thread-güvenli servis.

Özellikler:
    - Ayrı daemon thread'inde çalışır, ana döngüyü dondurmaz.
    - ``threading.Queue`` ile thread-güvenli mesaj iletişimi.
    - Cooldown mekanizması: aynı kategorideki uyarılar arasında minimum
      süre geçmesini zorunlu kılar (varsayılan 4 saniye).
    - Windows ve Linux uyumlu (``pyttsx3`` tabanlı TTS).

Tipik Kullanım:
    >>> from services.alert_service import AlertService
    >>>
    >>> alert = AlertService(cooldown_seconds=4.0)
    >>> alert.start()
    >>> alert.trigger("drowsy")     # "Lütfen uyanın!"
    >>> alert.trigger("yawning")    # "Esneme tespit edildi!"
    >>> alert.trigger("distracted") # "Yola odaklanın!"
    >>> alert.shutdown()
"""

from __future__ import annotations

import logging
import threading
import time
from queue import Empty, Queue
from typing import Final

import pyttsx3

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Sabit Tanımlamalar
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_COOLDOWN: Final[float] = 4.0
"""Aynı kategorideki iki uyarı arasındaki minimum süre (saniye)."""

_ALERT_MESSAGES: Final[dict[str, str]] = {
    "drowsy":     "Lütfen uyanın! Göz kapaklarınız kapanıyor.",
    "yawning":    "Esneme tespit edildi! Mola vermeyi düşünün.",
    "distracted": "Yola odaklanın! Dikkatiniz dağılmış görünüyor.",
    "phone":      "Lütfen sürüş esnasında telefon kullanmayın!",
}
"""Durum kategorilerine karşılık gelen Türkçe uyarı metinleri."""

_SENTINEL: Final[str] = "__SHUTDOWN__"
"""Worker thread'e kapatma sinyali göndermek için kullanılan sentinel değer."""


class AlertService:
    """Thread güvenli arka plan sesli uyarı servisi.

    Ana uygulama thread'ini bloke etmeden ayrı bir daemon thread'inde
    ``pyttsx3`` üzerinden Türkçe sesli ikazlar çalar. Her uyarı kategorisi
    için bağımsız cooldown takibi yapar.

    Attributes:
        _cooldown: Aynı kategori için minimum bekleme süresi (saniye).
        _queue: Thread-güvenli mesaj kuyruğu.
        _last_trigger_times: Kategori bazlı son tetikleme zamanları.
        _worker_thread: Arka plan TTS worker thread'i.
        _running: Servisin çalışıp çalışmadığını belirten bayrak.
        _lock: ``_last_trigger_times`` erişimi için kilit.
    """

    def __init__(self, cooldown_seconds: float = _DEFAULT_COOLDOWN) -> None:
        """AlertService nesnesini başlatır.

        Args:
            cooldown_seconds: Aynı kategorideki iki uyarı arasındaki minimum
                bekleme süresi (saniye). Varsayılan 4.0.

        Raises:
            ValueError: ``cooldown_seconds`` negatifse.
        """
        if cooldown_seconds < 0:
            raise ValueError(
                f"Cooldown süresi negatif olamaz: {cooldown_seconds}"
            )

        self._cooldown: float = cooldown_seconds
        self._queue: Queue[str] = Queue()
        self._last_trigger_times: dict[str, float] = {}
        self._lock: threading.Lock = threading.Lock()
        self._worker_thread: threading.Thread | None = None
        self._running: bool = False

    # ─────────────────────────────────────────────────────────────────────
    #  Yaşam Döngüsü
    # ─────────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Arka plan worker thread'ini başlatır.

        Birden fazla kez çağrılması güvenlidir; zaten çalışıyorsa
        hiçbir şey yapmaz.
        """
        if self._running:
            logger.debug("AlertService zaten çalışıyor, yeniden başlatma atlanıyor.")
            return

        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="AlertService-Worker",
            daemon=True,
        )
        self._worker_thread.start()
        logger.info("AlertService başlatıldı.")

    def shutdown(self, timeout: float = 5.0) -> None:
        """Servisi düzgün bir şekilde kapatır.

        Kuyruğa sentinel mesaj göndererek worker thread'in döngüsünden
        çıkmasını sağlar ve thread'in sonlanmasını bekler.

        Args:
            timeout: Worker thread'in sonlanması için bekleme süresi (saniye).
        """
        if not self._running:
            return

        self._running = False
        self._queue.put(_SENTINEL)

        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)
            if self._worker_thread.is_alive():
                logger.warning(
                    "AlertService worker thread %s saniyede sonlanmadı.",
                    timeout,
                )

        logger.info("AlertService kapatıldı.")

    # ─────────────────────────────────────────────────────────────────────
    #  Dışa Açık API
    # ─────────────────────────────────────────────────────────────────────

    def trigger(self, category: str) -> bool:
        """Belirtilen kategoride sesli uyarı tetikler.

        Cooldown süresi dolmamışsa uyarı atlanır ve ``False`` döner.

        Args:
            category: Uyarı kategorisi. Geçerli değerler:
                ``"drowsy"``, ``"yawning"``, ``"distracted"``.

        Returns:
            ``True`` uyarı kuyruğa eklendiyse, ``False`` cooldown
            nedeniyle atlandıysa.
        """
        if not self._running:
            logger.warning("AlertService çalışmıyor, trigger yok sayıldı.")
            return False

        if category not in _ALERT_MESSAGES:
            logger.warning("Bilinmeyen uyarı kategorisi: %s", category)
            return False

        now = time.monotonic()

        with self._lock:
            last_time = self._last_trigger_times.get(category, 0.0)
            if (now - last_time) < self._cooldown:
                logger.debug(
                    "'%s' uyarısı cooldown içinde, atlanıyor (%.1f sn kaldı).",
                    category,
                    self._cooldown - (now - last_time),
                )
                return False
            self._last_trigger_times[category] = now

        self._queue.put(category)
        logger.debug("'%s' uyarısı kuyruğa eklendi.", category)
        return True

    @property
    def is_running(self) -> bool:
        """Servisin aktif olup olmadığını döndürür."""
        return self._running

    # ─────────────────────────────────────────────────────────────────────
    #  Worker Thread
    # ─────────────────────────────────────────────────────────────────────

    def _worker_loop(self) -> None:
        """Arka plan thread'inde çalışan ana döngü.

        Kuyruktan mesaj alır ve ``pyttsx3`` ile sesli uyarı çalar.
        ``_SENTINEL`` mesajı alındığında döngüden çıkar.
        """
        engine: pyttsx3.Engine | None = None
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)

            # Türkçe ses varsa seçmeye çalış
            voices = engine.getProperty("voices")
            for voice in voices:
                if "turkish" in voice.name.lower() or "tr" in voice.id.lower():
                    engine.setProperty("voice", voice.id)
                    logger.info("Türkçe TTS sesi seçildi: %s", voice.name)
                    break

        except Exception:
            logger.exception("pyttsx3 başlatılamadı, sesli uyarılar devre dışı.")
            self._running = False
            return

        while self._running:
            try:
                category = self._queue.get(timeout=0.5)
            except Empty:
                continue

            if category == _SENTINEL:
                break

            message = _ALERT_MESSAGES.get(category)
            if message is None:
                continue

            try:
                logger.info("Sesli uyarı çalınıyor: %s", message)
                engine.say(message)
                engine.runAndWait()
            except Exception:
                logger.exception("Sesli uyarı çalınamadı: %s", message)

        # Motor kaynaklarını temizle
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────────
    #  Context Manager Desteği
    # ─────────────────────────────────────────────────────────────────────

    def __enter__(self) -> AlertService:
        """Context manager giriş: servisi başlatır."""
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager çıkış: servisi kapatır."""
        self.shutdown()
