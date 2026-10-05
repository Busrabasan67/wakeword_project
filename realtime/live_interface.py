"""
Hey Pakize – Modern Live Interface (PyQt5 + sounddevice)
Delta MFCC based real-time wake-word detection.

Installation:
    pip install pyqt5 sounddevice

Usage:
    python -m realtime.live_interface
    python -m realtime.live_interface models/wakeword_model.joblib
"""
from __future__ import annotations

import math
import queue
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import joblib

try:
    import sounddevice as sd  # type: ignore[import-untyped]
    SOUNDDEVICE_OK = True
except ImportError:
    sd = None  # type: ignore[assignment]
    SOUNDDEVICE_OK = False

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget,
        QVBoxLayout, QHBoxLayout, QLabel,
        QProgressBar, QFrame, QPushButton,
        QSlider, QListWidget, QListWidgetItem,
        QSizePolicy, QGraphicsOpacityEffect,
    )
    from PyQt5.QtCore import (
        Qt, QTimer, QObject, pyqtSignal,
        QPropertyAnimation, QEasingCurve, QRectF,
    )
    from PyQt5.QtGui import (
        QFont, QPainter, QPen, QColor,
        QLinearGradient, QPainterPath, QBrush,
        QRadialGradient,
    )
    PYQT5_OK = True
except ImportError:
    PYQT5_OK = False
    QObject = object  # type: ignore[misc,assignment]
    pyqtSignal = None  # type: ignore[assignment]

from audio.preprocess import normalize_volume, pad_or_crop, DEFAULT_SR, TARGET_DURATION_SEC
from audio.feature_extraction import extract_mfcc_features

# ── Sabitler ──────────────────────────────────────────────────────────────────
CHUNK = 1024
BUFFER_SEC = 2.0                               # 2 saniyelik kayar pencere
BUFFER_SIZE = int(DEFAULT_SR * BUFFER_SEC)
WINDOW_SIZE = int(DEFAULT_SR * TARGET_DURATION_SEC)  # model girişi: 1 sn
SLIDE_STEP  = int(DEFAULT_SR * 0.10)           # 100ms adımlarla tara
HOP_MS = 300
MIN_RMS_TRIGGER = 0.01
COOLDOWN_SEC = 1.5
MAX_LOG = 50

# Renk paleti
C_BG        = "#080C14"
C_PANEL     = "#0F1523"
C_BORDER    = "#1E2D45"
C_CYAN      = "#00D4FF"
C_PURPLE    = "#8B5CF6"
C_GREEN     = "#10B981"
C_RED       = "#EF4444"
C_TEXT      = "#E2E8F0"
C_MUTED     = "#64748B"
C_MUTED2    = "#1E2D45"


# ── Waveform Widget ───────────────────────────────────────────────────────────
class WaveformWidget(QWidget):
    """Canlı ses dalgasını QPainter ile çizer."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(80)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._samples: np.ndarray = np.zeros(512, dtype=np.float32)
        self._color = QColor(C_CYAN)

    def update_samples(self, samples: np.ndarray) -> None:
        n = min(len(samples), 512)
        self._samples = samples[-n:].copy() if n > 0 else np.zeros(512, dtype=np.float32)
        self.update()

    def set_active(self, active: bool) -> None:
        self._color = QColor(C_GREEN if active else C_CYAN)
        self.update()

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        mid = h // 2

        # Arka plan
        p.fillRect(0, 0, w, h, QColor(C_PANEL))

        # Alt çizgi (baseline)
        pen = QPen(QColor(C_BORDER), 1)
        p.setPen(pen)
        p.drawLine(0, mid, w, mid)

        samples = self._samples
        if len(samples) < 2:
            return

        # Gradyan renk
        grad = QLinearGradient(0, 0, w, 0)
        col = self._color
        col_fade = QColor(col)
        col_fade.setAlpha(60)
        grad.setColorAt(0.0, col_fade)
        grad.setColorAt(0.5, col)
        grad.setColorAt(1.0, col_fade)

        pen = QPen(QBrush(grad), 1.5)
        p.setPen(pen)

        path = QPainterPath()
        step = w / (len(samples) - 1)
        amp = (h // 2 - 6)
        path.moveTo(0, mid - float(samples[0]) * amp)
        for i, s in enumerate(samples[1:], 1):
            path.lineTo(i * step, mid - float(s) * amp)
        p.drawPath(path)


# ── Nabız Halkası ─────────────────────────────────────────────────────────────
class PulseRing(QWidget):
    """Algılama sırasında nabız atan animasyonlu daire."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(200, 200)
        self._pulse_radius: float = 0.0
        self._pulse_alpha: float = 0.0
        self._active = False
        self._prob: float = 0.0

        self._anim_timer = QTimer()
        self._anim_timer.timeout.connect(self._tick)
        self._anim_timer.start(30)

    def set_prob(self, prob: float) -> None:
        self._prob = prob
        self.update()

    def set_active(self, active: bool) -> None:
        if active and not self._active:
            self._pulse_radius = 0.0
            self._pulse_alpha = 1.0
        self._active = active
        self.update()

    def _tick(self) -> None:
        if self._pulse_alpha > 0:
            self._pulse_radius += 3.0
            self._pulse_alpha = max(0.0, self._pulse_alpha - 0.04)
            self.update()

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        cx, cy = self.width() // 2, self.height() // 2
        base_r = 72

        # Yayılan halka (pulse)
        if self._pulse_alpha > 0:
            r = base_r + self._pulse_radius
            col = QColor(C_GREEN if self._active else C_CYAN)
            col.setAlphaF(self._pulse_alpha * 0.6)
            pen = QPen(col, 2)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(int(cx - r), int(cy - r), int(r * 2), int(r * 2))

        # Prob dolum yayı
        arc_col = QColor(C_GREEN if self._active else C_CYAN)
        arc_col.setAlphaF(0.25)
        pen = QPen(arc_col, 10)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        rect = QRectF(cx - base_r + 5, cy - base_r + 5, (base_r - 5) * 2, (base_r - 5) * 2)
        p.drawArc(rect, 90 * 16, -360 * 16)

        # Dolum yayı (prob miktarına göre)
        if self._prob > 0.01:
            arc_col2 = QColor(C_GREEN if self._active else C_CYAN)
            pen2 = QPen(arc_col2, 10)
            pen2.setCapStyle(Qt.RoundCap)
            p.setPen(pen2)
            span = int(-self._prob * 360 * 16)
            p.drawArc(rect, 90 * 16, span)

        # İç dolgu
        inner_r = base_r - 18
        if self._active:
            grad = QRadialGradient(cx, cy, inner_r)
            grad.setColorAt(0.0, QColor(C_GREEN).lighter(130))
            grad.setColorAt(1.0, QColor(C_GREEN))
            p.setBrush(QBrush(grad))
        else:
            p.setBrush(QColor(C_PANEL))
        p.setPen(QPen(QColor(C_BORDER), 2))
        p.drawEllipse(cx - inner_r, cy - inner_r, inner_r * 2, inner_r * 2)

        # İkon
        p.setFont(QFont("Segoe UI", 22))
        icon = "🔔" if self._active else "🎙"
        p.setPen(QColor(C_TEXT))
        p.drawText(QRectF(cx - 30, cy - 20, 60, 40), Qt.AlignCenter, icon)


# ── Yatay Metre ───────────────────────────────────────────────────────────────
class MeterRow(QWidget):
    """Etiket + özel progress bar tek satırda."""

    def __init__(self, label: str, color: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._color = color
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        lbl = QLabel(label)
        lbl.setFixedWidth(80)
        lbl.setFont(QFont("Segoe UI", 9))
        lbl.setStyleSheet(f"color: {C_MUTED};")
        lay.addWidget(lbl)

        self.bar = QProgressBar()
        self.bar.setMinimum(0)
        self.bar.setMaximum(1000)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(10)
        self.bar.setStyleSheet(f"""
            QProgressBar {{
                background: {C_MUTED2};
                border-radius: 5px;
            }}
            QProgressBar::chunk {{
                background: {color};
                border-radius: 5px;
            }}
        """)
        lay.addWidget(self.bar)

        self.val_lbl = QLabel("0%")
        self.val_lbl.setFixedWidth(36)
        self.val_lbl.setFont(QFont("Segoe UI", 9))
        self.val_lbl.setStyleSheet(f"color: {color};")
        self.val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lay.addWidget(self.val_lbl)

    def set_value(self, v: float) -> None:
        iv = int(v * 1000)
        self.bar.setValue(min(iv, 1000))
        self.val_lbl.setText(f"{int(v * 100)}%")


# ── Separator ─────────────────────────────────────────────────────────────────
def _sep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setStyleSheet(f"background: {C_BORDER}; border: none; max-height: 1px;")
    return f


# ── Audio Worker ──────────────────────────────────────────────────────────────
class AudioWorker(QObject):
    if PYQT5_OK:
        detection_signal = pyqtSignal(float, float)
        error_signal = pyqtSignal(str)

    def __init__(
        self,
        model,
        threshold: float,
        wav_queue: "queue.Queue[np.ndarray]",
        n_mfcc: int = 20,
    ) -> None:
        if PYQT5_OK:
            super().__init__()
        self.model = model
        self.threshold = threshold
        self._wav_queue = wav_queue
        self._n_mfcc = n_mfcc
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not SOUNDDEVICE_OK:
            if PYQT5_OK:
                self.error_signal.emit("sounddevice not found.")
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        try:
            ring: deque[float] = deque(maxlen=BUFFER_SIZE)

            def sd_callback(indata: np.ndarray, frames: int, _t, _s) -> None:
                ring.extend(indata[:, 0].tolist())
                # Waveform kuyruğuna son 512 örneği koy (sıkışırsa atla)
                if not self._wav_queue.full():
                    arr = np.array(list(ring)[-512:], dtype=np.float32)
                    self._wav_queue.put_nowait(arr)

            with sd.InputStream(  # type: ignore[union-attr]
                channels=1,
                samplerate=DEFAULT_SR,
                dtype="float32",
                blocksize=CHUNK,
                callback=sd_callback,
            ):
                last_emit = 0.0
                while self._running:
                    sd.sleep(int(HOP_MS))  # type: ignore[union-attr]
                    now = time.time()
                    if now - last_emit < HOP_MS / 1000.0:
                        continue
                    if len(ring) < WINDOW_SIZE:
                        continue
                    buf = np.array(ring, dtype=np.float32)
                    last_emit = now

                    # Top-3 RMS penceresi → MAX prob inference
                    # ─────────────────────────────────────────────────────
                    # 2 sn buffer'da 1 sn pencereyi 100ms adımlarla kaydır.
                    # Tüm pencereleri RMS'e göre sırala, en enerjik 3'ünü al.
                    # Bu 3 pencere büyük ihtimalle phrasi içerir.
                    # Sadece bu 3'ün MAX prob'unu döndür:
                    #   • Max-of-all'dan daha az false positive (tüm pencereler değil)
                    #   • Weighted-avg'den daha iyi detection (ortalama değil max)
                    n_steps = max(1, (len(buf) - WINDOW_SIZE) // SLIDE_STEP + 1)
                    candidates: list[tuple[float, np.ndarray]] = []

                    for k in range(n_steps):
                        s = k * SLIDE_STEP
                        e = s + WINDOW_SIZE
                        if e > len(buf):
                            break
                        cand = buf[s:e]
                        cand_rms = float(np.sqrt(np.mean(cand ** 2) + 1e-12))
                        if cand_rms >= MIN_RMS_TRIGGER:
                            candidates.append((cand_rms, cand.copy()))

                    # En enerjik 3 pencereye inference uygula
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    top_k = candidates[:3]

                    best_prob = 0.0
                    max_rms = top_k[0][0] if top_k else 0.0
                    for _, cand in top_k:
                        sig = normalize_volume(cand.copy())
                        sig = pad_or_crop(sig, sr=DEFAULT_SR, target_duration=TARGET_DURATION_SEC)
                        feats = extract_mfcc_features(
                            sig, sr=DEFAULT_SR, n_mfcc=self._n_mfcc, use_delta=True
                        )
                        p = float(self.model.predict_proba(feats.reshape(1, -1))[0, 1])
                        if p > best_prob:
                            best_prob = p

                    prob = best_prob
                    rms = max_rms if max_rms > 0 else float(
                        np.sqrt(np.mean(buf[-WINDOW_SIZE:] ** 2) + 1e-12)
                    )
                    if PYQT5_OK:
                        self.detection_signal.emit(prob, rms)
        except Exception as exc:
            if PYQT5_OK:
                self.error_signal.emit(str(exc))


# ── Ana Pencere ───────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self, model_path: str | Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Hey Pakize  ·  Wake-Word Detector")
        self.setMinimumSize(620, 860)
        if model_path is None:
            model_path = _project_root / "models" / "wakeword_model.joblib"
        model_path_str = str(model_path)
        self._load_model(model_path_str)
        self._model_path = Path(model_path_str).name
        self._detection_count = 0
        self._last_trigger_time = 0.0
        self._wav_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=8)
        self._worker: AudioWorker | None = None
        self._glow_timer = QTimer()
        self._glow_timer.timeout.connect(self._reset_active)
        self._wav_timer = QTimer()
        self._wav_timer.timeout.connect(self._drain_waveform)
        self._build_ui()

    # ── Model yükle ───────────────────────────────────────────────────────
    def _load_model(self, path: str) -> None:
        try:
            payload = joblib.load(path)
            self.model = payload["model"]
            self.threshold = float(payload.get("prob_threshold", 0.80))
            # n_mfcc: model dosyasından oku; yoksa feature_cols sayısından çıkar
            feature_cols = payload.get("feature_cols", [])
            self.n_mfcc: int = int(
                payload.get("n_mfcc", max(1, len(feature_cols) // 6))
            )
        except Exception:
            self.model = None
            self.threshold = 0.80
            self.n_mfcc = 20

    # ── UI inşa ───────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        self.setStyleSheet(f"""
            QMainWindow, QWidget  {{ background: {C_BG}; color: {C_TEXT}; }}
            QListWidget           {{ background: {C_PANEL}; border: 1px solid {C_BORDER};
                                     border-radius: 8px; outline: 0; }}
            QListWidget::item     {{ padding: 5px 10px; border-bottom: 1px solid {C_BORDER}; }}
            QListWidget::item:selected {{ background: {C_BORDER}; }}
            QScrollBar:vertical   {{ background: {C_PANEL}; width: 6px; }}
            QScrollBar::handle:vertical {{ background: {C_BORDER}; border-radius: 3px; }}
        """)

        root = QWidget()
        self.setCentralWidget(root)
        main = QVBoxLayout(root)
        main.setContentsMargins(24, 20, 24, 16)
        main.setSpacing(0)

        # ── Başlık çubuğu
        main.addWidget(self._build_header())
        main.addSpacing(4)
        main.addWidget(_sep())
        main.addSpacing(14)

        # ── Merkez: nabız halkası
        pulse_row = QHBoxLayout()
        pulse_row.addStretch()
        self.pulse = PulseRing()
        pulse_row.addWidget(self.pulse)
        pulse_row.addStretch()
        main.addLayout(pulse_row)
        main.addSpacing(6)

        # Durum etiketi
        self.status_lbl = QLabel("Press the button to start listening")
        self.status_lbl.setFont(QFont("Segoe UI", 13, QFont.Bold))
        self.status_lbl.setAlignment(Qt.AlignCenter)
        self.status_lbl.setStyleSheet(f"color: {C_MUTED};")
        main.addWidget(self.status_lbl)
        main.addSpacing(16)

        # ── Waveform
        wave_frame = QFrame()
        wave_frame.setStyleSheet(f"""
            QFrame {{ background: {C_PANEL}; border: 1px solid {C_BORDER};
                      border-radius: 10px; }}
        """)
        wf_lay = QVBoxLayout(wave_frame)
        wf_lay.setContentsMargins(10, 8, 10, 8)
        wf_lay.setSpacing(4)
        wlbl = QLabel("SOUND WAVE")
        wlbl.setFont(QFont("Segoe UI", 8))
        wlbl.setStyleSheet(f"color: {C_MUTED}; letter-spacing: 2px;")
        wf_lay.addWidget(wlbl)
        self.waveform = WaveformWidget()
        wf_lay.addWidget(self.waveform)
        main.addWidget(wave_frame)
        main.addSpacing(14)

        # ── Metre paneli
        meter_frame = self._panel()
        mf_lay = QVBoxLayout(meter_frame)
        mf_lay.setContentsMargins(16, 12, 16, 12)
        mf_lay.setSpacing(8)
        self.prob_meter = MeterRow("Probability", C_CYAN)
        self.rms_meter = MeterRow("Audio (RMS)", C_PURPLE)
        mf_lay.addWidget(self.prob_meter)
        mf_lay.addWidget(self.rms_meter)
        mf_lay.addSpacing(4)

        # Eşik slider
        thresh_row = QHBoxLayout()
        tl = QLabel("Threshold")
        tl.setFont(QFont("Segoe UI", 9))
        tl.setStyleSheet(f"color: {C_MUTED};")
        tl.setFixedWidth(80)
        thresh_row.addWidget(tl)
        self.thresh_slider = QSlider(Qt.Horizontal)
        self.thresh_slider.setMinimum(50)
        self.thresh_slider.setMaximum(95)
        self.thresh_slider.setValue(int(round(self.threshold * 100)))
        self.thresh_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                background: {C_MUTED2}; height: 6px; border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {C_CYAN}; width: 16px; margin: -5px 0; border-radius: 8px;
            }}
            QSlider::sub-page:horizontal {{
                background: {C_CYAN}; border-radius: 3px;
            }}
        """)
        self.thresh_slider.valueChanged.connect(self._on_thresh_changed)
        thresh_row.addWidget(self.thresh_slider)
        self.thresh_val_lbl = QLabel(f"{self.threshold:.2f}")
        self.thresh_val_lbl.setFixedWidth(36)
        self.thresh_val_lbl.setFont(QFont("Segoe UI", 9))
        self.thresh_val_lbl.setStyleSheet(f"color: {C_CYAN};")
        self.thresh_val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        thresh_row.addWidget(self.thresh_val_lbl)
        mf_lay.addLayout(thresh_row)
        main.addWidget(meter_frame)
        main.addSpacing(10)

        # ── Stats çubuğu
        stats_frame = self._panel()
        sf_lay = QHBoxLayout(stats_frame)
        sf_lay.setContentsMargins(16, 10, 16, 10)

        model_ok = self.model is not None
        model_color = C_GREEN if model_ok else C_RED
        model_icon = "●" if model_ok else "●"
        n_mfcc_str = f"  ·  MFCC={self.n_mfcc}" if model_ok else ""
        self.model_lbl = QLabel(
            f"<span style='color:{model_color};'>{model_icon}</span>"
            f"  {self._model_path}{n_mfcc_str}"
        )
        self.model_lbl.setFont(QFont("Segoe UI", 9))
        self.model_lbl.setStyleSheet(f"color: {C_MUTED};")
        sf_lay.addWidget(self.model_lbl)
        sf_lay.addStretch()
        self.count_lbl = QLabel("0 detections")
        self.count_lbl.setFont(QFont("Segoe UI", 9))
        self.count_lbl.setStyleSheet(f"color: {C_MUTED};")
        sf_lay.addWidget(self.count_lbl)
        sf_lay.addSpacing(16)
        self.last_lbl = QLabel("Last: —")
        self.last_lbl.setFont(QFont("Segoe UI", 9))
        self.last_lbl.setStyleSheet(f"color: {C_MUTED};")
        sf_lay.addWidget(self.last_lbl)
        main.addWidget(stats_frame)
        main.addSpacing(10)

        # ── Geçmiş log
        log_frame = self._panel()
        lf_lay = QVBoxLayout(log_frame)
        lf_lay.setContentsMargins(16, 10, 16, 10)
        lf_lay.setSpacing(6)
        log_hdr = QHBoxLayout()
        log_title = QLabel("DETECTION HISTORY")
        log_title.setFont(QFont("Segoe UI", 8))
        log_title.setStyleSheet(f"color: {C_MUTED}; letter-spacing: 2px;")
        log_hdr.addWidget(log_title)
        log_hdr.addStretch()
        self.log_count_lbl = QLabel("last 50")
        self.log_count_lbl.setFont(QFont("Segoe UI", 8))
        self.log_count_lbl.setStyleSheet(f"color: {C_BORDER};")
        log_hdr.addWidget(self.log_count_lbl)
        lf_lay.addLayout(log_hdr)
        self.log_list = QListWidget()
        self.log_list.setFixedHeight(110)
        self.log_list.setFont(QFont("Segoe UI Mono", 9))
        lf_lay.addWidget(self.log_list)
        main.addWidget(log_frame)
        main.addSpacing(14)

        # ── Başlat butonu
        self.btn = QPushButton("▶   START LISTENING")
        self.btn.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.btn.setMinimumHeight(52)
        self.btn.setCursor(Qt.PointingHandCursor)
        self._set_btn_style(active=False)
        self.btn.clicked.connect(self._toggle)
        main.addWidget(self.btn)

    def _build_header(self) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)

        title = QLabel("◆  HEY PAKİZE")
        title.setFont(QFont("Segoe UI", 20, QFont.Bold))
        title.setStyleSheet(f"color: {C_CYAN}; letter-spacing: 3px;")
        lay.addWidget(title)
        lay.addStretch()

        sub = QLabel("Wake-Word Detector  ·  SVM + Δ-MFCC")
        sub.setFont(QFont("Segoe UI", 9))
        sub.setStyleSheet(f"color: {C_MUTED};")
        sub.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lay.addWidget(sub)
        return w

    def _panel(self) -> QFrame:
        f = QFrame()
        f.setStyleSheet(f"""
            QFrame {{
                background: {C_PANEL};
                border: 1px solid {C_BORDER};
                border-radius: 10px;
            }}
        """)
        return f

    def _set_btn_style(self, active: bool) -> None:
        if active:
            self.btn.setStyleSheet(f"""
                QPushButton {{
                    background: {C_RED};
                    color: #fff;
                    border: none;
                    border-radius: 12px;
                    letter-spacing: 1px;
                }}
                QPushButton:hover {{ background: #dc2626; }}
            """)
        else:
            self.btn.setStyleSheet(f"""
                QPushButton {{
                    background: {C_GREEN};
                    color: #0A0A14;
                    border: none;
                    border-radius: 12px;
                    letter-spacing: 1px;
                }}
                QPushButton:hover {{ background: #059669; }}
            """)

    # ── Slider ────────────────────────────────────────────────────────────
    def _on_thresh_changed(self, v: int) -> None:
        self.threshold = v / 100.0
        self.thresh_val_lbl.setText(f"{self.threshold:.2f}")

    # ── Başlat / Durdur ───────────────────────────────────────────────────
    def _toggle(self) -> None:
        if self._worker and self._worker._running:
            self._stop_listening()
        else:
            self._start_listening()

    def _start_listening(self) -> None:
        if self.model is None:
            self._log_entry("Model not loaded!", C_RED)
            return
        self._worker = AudioWorker(self.model, self.threshold, self._wav_queue, self.n_mfcc)
        self._worker.detection_signal.connect(self._on_detection)
        self._worker.error_signal.connect(lambda e: self._log_entry(f"Error: {e}", C_RED))
        self._worker.start()
        self._wav_timer.start(50)
        self.btn.setText("■   STOP")
        self._set_btn_style(active=True)
        self.status_lbl.setText("Listening…")
        self.status_lbl.setStyleSheet(f"color: {C_MUTED};")
        self.waveform.set_active(False)

    def _stop_listening(self) -> None:
        if self._worker:
            self._worker.stop()
        self._wav_timer.stop()
        self.btn.setText("▶   START LISTENING")
        self._set_btn_style(active=False)
        self.status_lbl.setText("Stopped")
        self.status_lbl.setStyleSheet(f"color: {C_MUTED};")
        self.waveform.update_samples(np.zeros(512))

    # ── Waveform kuyruğunu boşalt (ana thread timer'ı) ────────────────────
    def _drain_waveform(self) -> None:
        arr = None
        while not self._wav_queue.empty():
            try:
                arr = self._wav_queue.get_nowait()
            except queue.Empty:
                break
        if arr is not None:
            self.waveform.update_samples(arr)

    # ── Algılama sinyali ──────────────────────────────────────────────────
    def _on_detection(self, prob: float, rms: float) -> None:
        self.prob_meter.set_value(prob)
        self.rms_meter.set_value(min(rms * 5, 1.0))
        self.pulse.set_prob(prob)

        now = time.time()
        thresh = self.thresh_slider.value() / 100.0
        should_trigger = (
            prob >= thresh
            and rms >= MIN_RMS_TRIGGER
            and (now - self._last_trigger_time) >= COOLDOWN_SEC
        )

        if should_trigger:
            self._last_trigger_time = now
            self._detection_count += 1
            ts = datetime.now().strftime("%H:%M:%S")
            self.pulse.set_active(True)
            self.waveform.set_active(True)
            self.status_lbl.setText("🔔  HEY PAKİZE!")
            self.status_lbl.setStyleSheet(
                f"color: {C_GREEN}; font-size: 16px; font-weight: bold;"
            )
            self.count_lbl.setText(f"{self._detection_count} detections")
            self.last_lbl.setText(f"Last: {ts}")
            self._log_entry(f"{ts}  ·  Hey Pakize detected  ({prob:.0%})", C_GREEN)
            self._glow_timer.start(1800)
        else:
            p_pct = int(prob * 100)
            if not self._glow_timer.isActive():
                self.status_lbl.setText(f"Listening…  ({p_pct}%)")
                self.status_lbl.setStyleSheet(f"color: {C_MUTED};")

    def _reset_active(self) -> None:
        self._glow_timer.stop()
        self.pulse.set_active(False)
        self.waveform.set_active(False)
        self.status_lbl.setText("Listening…")
        self.status_lbl.setStyleSheet(f"color: {C_MUTED};")

    def _log_entry(self, text: str, color: str = C_TEXT) -> None:
        item = QListWidgetItem(text)
        item.setForeground(QColor(color))
        self.log_list.insertItem(0, item)
        if self.log_list.count() > MAX_LOG:
            self.log_list.takeItem(self.log_list.count() - 1)
        self.log_count_lbl.setText(f"last {min(self.log_list.count(), MAX_LOG)}")

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._worker:
            self._worker.stop()
        event.accept()


# ── Giriş ─────────────────────────────────────────────────────────────────────
def main() -> int:
    if not PYQT5_OK:
        print("[!] PyQt5 not found. Install: pip install pyqt5")
        return 1

    default_model_path = str(_project_root / "models" / "wakeword_model.joblib")
    model_path = sys.argv[1] if len(sys.argv) > 1 else default_model_path
    if not Path(model_path).exists():
        print(f"[!] Model not found: {model_path}")

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow(model_path)
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
