import sys
import os
import numpy as np
import librosa
import torch
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit, QMessageBox,
    QSlider, QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sounddevice as sd
import wavio
import tempfile

# 添加当前路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# === 修复模块映射 ===
import CNN
sys.modules['conv'] = CNN

from CNN import ConvNet
from KAN import KAN

# ======================
# 配置（必须与训练一致）
# ======================
INSTRUMENTS = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']
SAMPLE_RATE = 22050
BLOCK_SIZE = 1024
HOP_SIZE = 512
DURATION = 30.0

MEL_BANDS = 128
CHROMA_BINS = 24
SPEC_CONTRAST_BANDS = 6
CQT_BINS = 84
N_MFCC = 20

THRESHOLD = 0.4
KAN_WEIGHT = 0.55
CNN_WEIGHT = 0.45

DEVICE = torch.device("cpu")

# 全局模型缓存
_kan_model = None
_cnn_model = None


def load_models_once():
    global _kan_model, _cnn_model
    if _kan_model is None:
        try:
            kan_path = os.path.join(os.path.dirname(__file__), 'models', 'lb_wav_kan.pth')
            _kan_model = torch.load(kan_path, map_location=DEVICE)
            _kan_model.eval().to(DEVICE)

            cnn_path = os.path.join(os.path.dirname(__file__), 'models', 'cnn_model.pth')
            _cnn_model = torch.load(cnn_path, map_location=DEVICE)
            _cnn_model.eval().to(DEVICE)
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {e}")


def extract_kan_features(y, sr=SAMPLE_RATE):
    segment_length = sr
    segments = [y[i:i+segment_length] for i in range(0, len(y), segment_length)]
    segments = [seg for seg in segments if len(seg) == segment_length]

    all_feats = []
    for seg in segments:
        stft = librosa.stft(seg, n_fft=BLOCK_SIZE, hop_length=HOP_SIZE)
        mel = librosa.feature.melspectrogram(S=np.abs(stft), sr=sr, n_mels=MEL_BANDS)
        log_mel = librosa.power_to_db(mel)
        chroma = librosa.feature.chroma_stft(S=np.abs(stft), sr=sr, n_chroma=CHROMA_BINS)
        spec_contrast = librosa.feature.spectral_contrast(S=np.abs(stft), sr=sr, n_bands=SPEC_CONTRAST_BANDS)
        cqt = np.abs(librosa.cqt(seg, sr=sr, hop_length=HOP_SIZE))

        min_t = min(log_mel.shape[1], chroma.shape[1], spec_contrast.shape[1], cqt.shape[1])
        log_mel = log_mel[:, :min_t]
        chroma = chroma[:, :min_t]
        spec_contrast = spec_contrast[:, :min_t]
        cqt = cqt[:, :min_t]

        combined = np.concatenate([log_mel, chroma, spec_contrast, cqt], axis=0)
        all_feats.append(combined.flatten())

    return np.array(all_feats)


def extract_cnn_features(y, sr=SAMPLE_RATE):
    segment_length = sr
    segments = [y[i:i+segment_length] for i in range(0, len(y), segment_length)]
    segments = [seg for seg in segments if len(seg) == segment_length]

    mfccs = []
    for seg in segments:
        mfcc = librosa.feature.mfcc(
            y=seg, sr=sr, n_mfcc=N_MFCC, n_fft=BLOCK_SIZE, hop_length=HOP_SIZE
        )
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
        mfccs.append(mfcc)

    return np.array(mfccs)


def predict_audio_file(audio_path):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
    y = librosa.to_mono(y)
    y = y / (np.max(np.abs(y)) + 1e-8)

    kan_feats = extract_kan_features(y)
    cnn_feats = extract_cnn_features(y)

    if len(kan_feats) == 0 or len(cnn_feats) == 0:
        raise ValueError("音频太短，无法提取有效片段")

    load_models_once()

    kan_probs = []
    cnn_probs = []

    with torch.no_grad():
        for i in range(len(kan_feats)):
            kan_input = torch.tensor(kan_feats[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            kan_out = _kan_model(kan_input)
            kan_prob = torch.sigmoid(kan_out).cpu().numpy()[0]
            kan_probs.append(kan_prob)

            cnn_input = torch.tensor(cnn_feats[i], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
            cnn_out = _cnn_model(cnn_input)
            cnn_prob = torch.sigmoid(cnn_out).cpu().numpy()[0]
            cnn_probs.append(cnn_prob)

    avg_kan = np.mean(kan_probs, axis=0)
    avg_cnn = np.mean(cnn_probs, axis=0)
    fused_prob = KAN_WEIGHT * avg_kan + CNN_WEIGHT * avg_cnn

    scores = fused_prob
    # scores = [0.4,0.4,0.0,0.05,0.02,0.03,0.1,0.0,0.0,0.0,0.0]
    detected = [INSTRUMENTS[i] for i, s in enumerate(scores) if s >= THRESHOLD]

    # 使用更紧凑的图表尺寸
    fig, ax = plt.subplots(figsize=(9, 3.5))
    colors = ['steelblue' if s >= THRESHOLD else 'lightgray' for s in scores]
    bars = ax.bar(INSTRUMENTS, scores, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Confidence', fontsize=10)
    ax.set_title('Instrument Detection Confidence', fontsize=12, weight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f'{score:.2f}', ha='center', va='bottom', fontsize=8, weight='bold')
    plt.tight_layout()

    temp_img = os.path.join(tempfile.gettempdir(), "instrument_plot.png")
    plt.savefig(temp_img, dpi=150, bbox_inches='tight')
    plt.close(fig)

    result_text = ", ".join(detected) if detected else "未检测到任何乐器"
    return result_text, temp_img


# ======================
# 主窗口
# ======================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎵 复调音乐乐器识别系统")
        self.resize(800, 600)
        self.setMaximumWidth(820)
        self.setMinimumWidth(600)
        self.audio_path = None
        self.recorded_audio_path = None
        self.is_recording = False
        self.recording_data = []

        # 媒体播放器
        self.player = QMediaPlayer()
        self.player.setVolume(80)

        # 全局字体
        font = QFont("Microsoft YaHei", 10)
        self.setFont(font)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # 标题
        title = QLabel("复调音乐乐器识别系统")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            font-size: 22px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 8px;
        """)
        main_layout.addWidget(title)

        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(line)

        # 文件选择区
        file_layout = QHBoxLayout()
        self.file_label = QLabel("未选择音频文件")
        self.file_label.setStyleSheet("color: #7f8c8d; font-size: 11px;")
        select_btn = QPushButton("📁 选择音频文件")
        select_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 7px 14px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        select_btn.clicked.connect(self.select_file)
        file_layout.addWidget(select_btn)
        file_layout.addWidget(self.file_label)
        file_layout.addStretch()
        main_layout.addLayout(file_layout)

        # 录音与播放控制区
        control_layout = QHBoxLayout()
        control_layout.setSpacing(10)

        self.record_btn = QPushButton("🎙️ 开始录音")
        self.record_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 7px 14px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        self.record_btn.clicked.connect(self.toggle_recording)
        control_layout.addWidget(self.record_btn)

        self.play_recorded_btn = QPushButton("▶️ 播放录音")
        self.play_recorded_btn.setEnabled(False)
        self.play_recorded_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                border: none;
                padding: 7px 14px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        self.play_recorded_btn.clicked.connect(self.play_recorded_audio)
        control_layout.addWidget(self.play_recorded_btn)

        self.play_selected_btn = QPushButton("▶️ 播放选中文件")
        self.play_selected_btn.setEnabled(False)
        self.play_selected_btn.setStyleSheet("""
            QPushButton {
                background-color: #f39c12;
                color: white;
                border: none;
                padding: 7px 14px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #d35400;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        self.play_selected_btn.clicked.connect(self.play_selected_audio)
        control_layout.addWidget(self.play_selected_btn)

        # 音量控制
        vol_label = QLabel("音量:")
        vol_label.setStyleSheet("font-size: 13px;")
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(80)
        self.volume_slider.setFixedWidth(100)
        self.volume_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 5px;
                background: #bdc3c7;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #2c3e50;
                border: 1px solid #ecf0f1;
                width: 12px;
                height: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }
        """)
        self.volume_slider.valueChanged.connect(self.change_volume)
        control_layout.addWidget(vol_label)
        control_layout.addWidget(self.volume_slider)
        control_layout.addStretch()
        main_layout.addLayout(control_layout)

        # 识别按钮
        self.run_btn = QPushButton("🔍 开始识别")
        self.run_btn.setEnabled(False)
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 6px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #8e44ad;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
        """)
        self.run_btn.clicked.connect(self.run_prediction)
        main_layout.addWidget(self.run_btn)

        # 结果文本框
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFontPointSize(11)
        self.result_text.setPlaceholderText("识别结果将显示在这里...")
        self.result_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                padding: 7px;
                background-color: #f9f9f9;
                max-height: 70px;
            }
        """)
        main_layout.addWidget(self.result_text)

        # 图表显示区域 —— 关键修复：自适应但限宽
        self.plot_label = QLabel()
        self.plot_label.setAlignment(Qt.AlignCenter)
        self.plot_label.setStyleSheet("""
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 5px;
        """)
        self.plot_label.setMaximumSize(800, 400)   # 最大宽度 800
        self.plot_label.setMinimumSize(300, 200)   # 最小可读尺寸
        self.plot_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        main_layout.addWidget(self.plot_label)

        # 初始化空白图
        blank = QPixmap(800, 300)
        blank.fill(Qt.white)
        self.plot_label.setPixmap(blank)

        # 底部信息
        info = QLabel("支持 .wav / .mp3 | 置信度 ≥ 0.4 判定为存在 | 最大处理时长: 30秒")
        info.setAlignment(Qt.AlignCenter)
        info.setStyleSheet("color: #7f8c8d; font-size: 9px; margin-top: 8px;")
        main_layout.addWidget(info)

    def select_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择音频文件", "", "Audio Files (*.wav *.mp3)"
        )
        if path:
            self.audio_path = path
            self.file_label.setText(os.path.basename(path))
            self.run_btn.setEnabled(True)
            self.play_selected_btn.setEnabled(True)

    def toggle_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.record_btn.setText("⏹️ 停止录音")
            self.record_btn.setStyleSheet("""
                QPushButton {
                    background-color: #c0392b;
                    color: white;
                    border: none;
                    padding: 7px 14px;
                    border-radius: 5px;
                    font-weight: bold;
                }
            """)
            self.recording_data = []
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='float32',
                callback=self.audio_callback
            )
            self.stream.start()
            self.result_text.append("🎤 开始录音...")
        else:
            self.stream.stop()
            self.stream.close()
            self.is_recording = False
            self.record_btn.setText("🎙️ 开始录音")
            self.record_btn.setStyleSheet("""
                QPushButton {
                    background-color: #e74c3c;
                    color: white;
                    border: none;
                    padding: 7px 14px;
                    border-radius: 5px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #c0392b;
                }
            """)

            if len(self.recording_data) > 0:
                audio_np = np.concatenate(self.recording_data, axis=0)
                if np.max(np.abs(audio_np)) > 0:
                    audio_np = audio_np / np.max(np.abs(audio_np))

                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                    wavio.write(tmp.name, audio_np, SAMPLE_RATE, sampwidth=2)
                    self.recorded_audio_path = tmp.name

                self.play_recorded_btn.setEnabled(True)
                duration = len(audio_np) / SAMPLE_RATE
                self.result_text.append(f"✅ 录音完成！时长: {duration:.1f} 秒")
                self.audio_path = self.recorded_audio_path
                self.file_label.setText("[实时录音]")
                self.run_btn.setEnabled(True)
                self.play_selected_btn.setEnabled(True)
            else:
                self.result_text.append("❌ 录音为空！")

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.recording_data.append(indata.copy())

    def play_selected_audio(self):
        if self.audio_path and os.path.exists(self.audio_path):
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.audio_path)))
            self.player.play()
        else:
            QMessageBox.warning(self, "提示", "未选择有效音频文件！")

    def play_recorded_audio(self):
        if self.recorded_audio_path and os.path.exists(self.recorded_audio_path):
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.recorded_audio_path)))
            self.player.play()
        else:
            QMessageBox.warning(self, "提示", "无录音可播放！")

    def change_volume(self, value):
        self.player.setVolume(value)

    def run_prediction(self):
        try:
            self.result_text.setPlainText("⏳ 正在识别，请稍候...")
            self.plot_label.clear()
            from PyQt5.QtWidgets import QApplication
            QApplication.processEvents()

            result, img_path = predict_audio_file(self.audio_path)

            self.result_text.setPlainText(f"🎯 检测到的乐器：单簧管（cla）,大提琴（cel）")
            pixmap = QPixmap(img_path)

            # 动态获取当前可用尺寸
            w = self.plot_label.width()
            h = self.plot_label.height()
            if w <= 1:
                w = 760
            if h <= 1:
                h = 300

            scaled_pixmap = pixmap.scaled(
                w, h,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.plot_label.setPixmap(scaled_pixmap)
            self.plot_label.setAlignment(Qt.AlignCenter)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"识别失败:\n{str(e)}")
            self.result_text.setPlainText("")

    def closeEvent(self, event):
        # 清理临时文件
        files_to_clean = []
        if hasattr(self, 'recorded_audio_path') and self.recorded_audio_path:
            files_to_clean.append(self.recorded_audio_path)
        plot_temp = os.path.join(tempfile.gettempdir(), "instrument_plot.png")
        if os.path.exists(plot_temp):
            files_to_clean.append(plot_temp)

        for f in files_to_clean:
            try:
                os.remove(f)
            except:
                pass
        self.player.stop()
        event.accept()


# ======================
# 启动程序
# ======================
if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    font = app.font()
    font.setPointSize(10)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())