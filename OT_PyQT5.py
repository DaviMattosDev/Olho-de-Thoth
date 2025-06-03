import cv2
import numpy as np
import mediapipe as mp
import subprocess
import json
import os
import tensorflow as tf
from tensorflow.keras.applications import (
    Xception, EfficientNetB0, InceptionV3, ResNet50
)
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
import warnings
warnings.filterwarnings('ignore')

# --------------------------
# Função 1: Obter número total de frames do vídeo
# --------------------------
def get_video_frame_count(video_path):
    ffprobe_path = os.path.join("ffmpeg", "bin", "ffprobe.exe")
    cmd = [
        ffprobe_path,
        '-v', 'error',
        '-select_streams', 'v:0',
        '-count_packets',
        '-show_entries', 'stream=nb_read_packets',
        '-of', 'default=nw=1',
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        output = result.stdout.decode()
        for line in output.splitlines():
            if "nb_read_packets" in line:
                return int(line.split("=")[1])
    except Exception as e:
        print(f"[ERRO] Não foi possível ler quantidade de frames: {e}")
    return 30  # fallback


# --------------------------
# Função 2: Extrair frames (agora com todos)
# --------------------------
def extract_frames(video_path, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        count += 1

    cap.release()
    return frames


# --------------------------
# Função 3: Detectar faces e piscadas
# --------------------------
mp_face_mesh = mp.solutions.face_mesh

def detect_faces_and_micro(frames, log_callback=None):
    face_issues = 0
    blink_count = 0

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:

        for idx, frame in enumerate(frames):
            if log_callback:
                log_callback(f"[INFO] Analisando rosto - Frame {idx + 1}/{len(frames)}")

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks is None:
                face_issues += 1
            else:
                landmarks = results.multi_face_landmarks[0].landmark
                left_eye = [landmarks[i] for i in [386, 385, 380, 374, 368, 387]]
                right_eye = [landmarks[i] for i in [159, 158, 153, 145, 144, 160]]

                def eye_aspect_ratio(eye):
                    A = abs(eye[1].y - eye[5].y)
                    B = abs(eye[2].y - eye[4].y)
                    C = abs(eye[0].x - eye[3].x)
                    return (A + B) / (2.0 * C)

                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
                if ear < 0.2:
                    blink_count += 1

    return {"face_issues": face_issues, "blink_count": blink_count}


# --------------------------
# Função 4: Estimar nitidez
# --------------------------
def estimate_blurriness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm


# --------------------------
# Função 5: Jitter entre frames
# --------------------------
def detect_frame_jitter(frames, log_callback=None):
    diffs = []
    for i in range(1, len(frames)):
        if log_callback:
            log_callback(f"[INFO] Analisando tremor - Frame {i + 1}/{len(frames)}")
        diff = cv2.absdiff(frames[i], frames[i - 1])
        mean_diff = diff.mean()
        diffs.append(mean_diff)
    return diffs


# --------------------------
# Função 6: FFT (frequência espacial)
# --------------------------
def analyze_fft(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    mean_magnitude = np.mean(magnitude_spectrum)
    return mean_magnitude


# --------------------------
# Função 7: Analisar metadados
# --------------------------
def analyze_metadata(video_path, log_callback=None):
    if log_callback:
        log_callback("[INFO] Lendo metadados do arquivo...")
    ffprobe_path = os.path.join("ffmpeg", "bin", "ffprobe.exe")
    if not os.path.isfile(ffprobe_path):
        raise FileNotFoundError(f"Arquivo ffprobe não encontrado em: {ffprobe_path}")

    cmd = [
        ffprobe_path,
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        metadata = json.loads(result.stdout)
    except json.JSONDecodeError:
        if log_callback:
            log_callback("[ERRO] Não foi possível ler os metadados do vídeo.")
        metadata = {}
    return metadata


# --------------------------
# Função 8: Carregar modelo treinado
# --------------------------

def load_model(model_name, log_callback=None):
    preprocess = None
    if model_name == "Xception":
        base_model = Xception(weights='imagenet', include_top=False)
        preprocess = xception_preprocess
    elif model_name == "EfficientNet":
        base_model = EfficientNetB0(weights='imagenet', include_top=False)
        preprocess = efficientnet_preprocess
    elif model_name == "Inception":
        base_model = InceptionV3(weights='imagenet', include_top=False)
        preprocess = inception_preprocess
    elif model_name == "ResNet":
        base_model = ResNet50(weights='imagenet', include_top=False)
        preprocess = resnet_preprocess
    else:
        raise ValueError(f"Modelo desconhecido: {model_name}")

    x = GlobalAveragePooling2D()(base_model.output)
    predictions = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model, preprocess


# Classificar frame com modelo escolhido
def classify_frame_with_ai(model, frame, preprocess_func):
    img = cv2.resize(frame, (299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_func(img_array)
    prediction = model.predict(img_array, verbose=0)
    fake_prob = prediction[0][1]
    return fake_prob


# --------------------------
# Função 9: Análise individual com um modelo
# --------------------------
def analyze_with_single_model(video_path, log_callback=None, model_type="Xception"):
    frames = extract_frames(video_path)

    model, preprocess = load_model(model_type, log_callback)
    ai_scores = []
    for i, frame in enumerate(frames):
        ai_score = classify_frame_with_ai(model, frame, preprocess)
        ai_scores.append(ai_score)
        if log_callback and i % 5 == 0:
            log_callback(f"[INFO] [{model_type}] Processando IA - Frame {i + 1}/{len(frames)}")
    avg_ai_score = np.mean(ai_scores)

    return {
        "model": model_type,
        "avg_ai_score": avg_ai_score
    }


# --------------------------
# Função 10: Análise completa com todos os modelos
# --------------------------
def analyze_with_all_models(video_path, log_callback=None):
    all_results = []

    models = ["Xception", "EfficientNet", "Inception", "ResNet"]

    # Primeiro extrair frames apenas uma vez
    if log_callback:
        log_callback("[INFO] Extraindo frames do vídeo...")

    frames = extract_frames(video_path)

    if log_callback:
        log_callback(f"[INFO] Extraídos {len(frames)} frames.")

    # Deteção de rostos (uma vez só)
    if log_callback:
        log_callback("[INFO] Analisando rostos e microcomportamentos...")
    face_data = detect_faces_and_micro(frames)

    # Nitidez média (uma vez só)
    blur_scores = [estimate_blurriness(frame) for frame in frames]
    avg_blur = np.mean(blur_scores)

    # Tremores (uma vez só)
    jitter_scores = detect_frame_jitter(frames)
    avg_jitter = np.mean(jitter_scores) if jitter_scores else 0

    # Frequências (uma vez só)
    fft_scores = [analyze_fft(frame) for frame in frames]
    avg_fft = np.mean(fft_scores)

    # Metadados (uma vez só)
    metadata = analyze_metadata(video_path, log_callback)

    # Agora analisa com todos os modelos
    ai_scores = []
    for model_type in models:
        result = analyze_with_single_model(video_path, log_callback, model_type)
        ai_scores.append(result["avg_ai_score"])
        all_results.append(result)

    avg_ai_score = np.mean(ai_scores)

    # Decisão final com base na média dos modelos
    score = 0
    if face_data['face_issues'] > 5:
        score += 2
    if face_data['blink_count'] < 2:
        score += 1
    if avg_blur < 100:
        score += 1
    if avg_jitter > 10:
        score += 1
    if avg_fft > 200:
        score += 1
    if avg_ai_score > 0.6:
        score += 2

    meta_text = json.dumps(metadata).lower()
    ai_keywords = ["google", "lavf", "ai", "synthetic", "fake"]
    found_keywords = [kw for kw in ai_keywords if kw in meta_text]

    if found_keywords:
        score += 2

    result = {
        "face_issues": face_data['face_issues'],
        "blink_count": face_data['blink_count'],
        "avg_blur": avg_blur,
        "avg_jitter": avg_jitter,
        "avg_fft": avg_fft,
        "avg_ai_score": avg_ai_score * 100,
        "score": score,
        "found_keywords": found_keywords,
        "deepfake": score >= 4,
        "metadata": metadata,
        "per_model": all_results
    }

    return result


# --------------------------
# INTERFACE GRÁFICA - PyQt5
# --------------------------

import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel,
    QTextEdit, QVBoxLayout, QWidget, QFileDialog, QMessageBox,
    QHBoxLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QTextCharFormat, QColor, QSyntaxHighlighter, QIcon, QTextCursor


class JsonHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._brushes = {
            "key": QTextCharFormat(),
            "string": QTextCharFormat(),
            "number": QTextCharFormat(),
            "boolean": QTextCharFormat(),
            "bracket": QTextCharFormat(),
        }
        self._brushes["key"].setForeground(QColor("#FFD700"))
        self._brushes["string"].setForeground(QColor("#00FF00"))
        self._brushes["number"].setForeground(QColor("#1E90FF"))
        self._brushes["boolean"].setForeground(QColor("#FF6347"))
        self._brushes["bracket"].setForeground(QColor("#FFFFFF"))

    def highlightBlock(self, text):
        import re
        patterns = [
            (r'"[^"]+":', "key"),
            (r'"[^"]*"', "string"),
            (r'\b\d+(\.\d+)?\b', "number"),
            (r'\b(true|false|null)\b', "boolean"),
            (r'[\{\}\[\]]', "bracket"),
        ]
        for pattern, key in patterns:
            for match in re.finditer(pattern, text):
                start, end = match.span()
                self.setFormat(start, end - start, self._brushes[key])


class AnalysisThread(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, video_path, analyze_func):
        super().__init__()
        self.video_path = video_path
        self.analyze_func = analyze_func

    def run(self):
        try:
            result = self.analyze_func(self.video_path, log_callback=lambda msg: self.log.emit(msg))
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class OlhoDeThothGUI(QMainWindow):
    def __init__(self, analyze_func):
        super().__init__()
        self.setWindowTitle("👁️‍🗨️ Olho de Thoth — Detector de Deepfakes")
        self.setGeometry(100, 100, 1000, 700)
        self.setStyleSheet("""
            background-color: #1e1e2f;
            color: #d9dcd6;
            font-family: 'Arial';
        """)

        self.analyze_func = analyze_func
        self.analysis_thread = None
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()

        title = QLabel("👁️‍🗨️ OLHO DE THOTH")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 32px; font-weight: bold; color: #d291bc;")
        layout.addWidget(title)

        subtitle = QLabel("Detector Avançado de Deepfakes")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("font-size: 18px; color: #aaaaaa;")
        layout.addWidget(subtitle)

        btn_layout = QHBoxLayout()
        self.select_button = QPushButton("📂 Selecionar Vídeo")
        self.select_button.clicked.connect(self.select_video)
        self.select_button.setStyleSheet(self.button_style())
        btn_layout.addWidget(self.select_button)

        self.analyze_button = QPushButton("🔍 Iniciar Análise")
        self.analyze_button.clicked.connect(self.start_analysis)
        self.analyze_button.setEnabled(False)
        self.analyze_button.setStyleSheet(self.button_style())
        btn_layout.addWidget(self.analyze_button)

        layout.addLayout(btn_layout)

        self.path_label = QLabel("Nenhum vídeo selecionado")
        self.path_label.setStyleSheet("color: #bbbbbb; margin-top: 5px;")
        layout.addWidget(self.path_label)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("""
            background-color: #2a2a3d;
            color: #e0e0e0;
            padding: 10px;
            font-family: Consolas;
            font-size: 12px;
            border: 1px solid #444;
        """)
        layout.addWidget(self.log_box)

        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        self.result_box.setStyleSheet("""
            background-color: #2a2a3d;
            color: #e0e0e0;
            padding: 10px;
            font-family: Consolas;
            font-size: 12px;
            border: 1px solid #444;
        """)
        self.highlighter = JsonHighlighter(self.result_box.document())
        layout.addWidget(self.result_box)
        self.result_box.hide()

        main_widget.setLayout(layout)

    def button_style(self):
        return """
            QPushButton {
                background-color: #5a2a7c;
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
                border: 1px solid #9a55ff;
            }
            QPushButton:hover {
                background-color: #7a3a9f;
            }
        """

    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Selecionar Vídeo",
            "", "Vídeos (*.mp4 *.avi *.mov *.mkv)"
        )
        if path:
            self.video_path = path
            self.path_label.setText(path)
            self.analyze_button.setEnabled(True)

    def start_analysis(self):
        self.log_box.clear()
        self.result_box.clear()
        self.result_box.hide()
        self.progress_bar.setValue(0)
        self.select_button.setEnabled(False)
        self.analyze_button.setEnabled(False)

        self.analysis_thread = AnalysisThread(self.video_path, self.analyze_func)
        self.analysis_thread.finished.connect(self.show_result)
        self.analysis_thread.error.connect(self.show_error)
        self.analysis_thread.log.connect(self.update_log)
        self.analysis_thread.start()

    def update_log(self, message):
        cursor = self.log_box.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(message + "\n")
        self.log_box.setTextCursor(cursor)
        self.log_box.ensureCursorVisible()

        # Atualiza a barra de progresso com base no log
        if "Processando IA" in message:
            try:
                frame_num = int(message.split("Frame ")[1].split("/")[0])
                total_frames = int(message.split("/")[1].split(")")[0])
                progress = int((frame_num / total_frames) * 100)
                self.progress_bar.setValue(progress)
            except Exception:
                pass

    def show_result(self, result):
        self.progress_bar.setValue(100)
        self.log_box.hide()
        self.result_box.show()

        per_model_info = "\n".join([
            f" - {r['model']}: {r['avg_ai_score'] * 100:.2f}% de chance de deepfake"
            for r in result.get("per_model", [])
        ])

        output = (
            f"👁️‍🗨️ RESULTADO DA ANÁLISE CONSOLIDADA\n"
            f"{'-'*40}\n"
            f"Problemas faciais detectados: {result['face_issues']}\n"
            f"Piscadelas detectadas: {result['blink_count']}\n"
            f"Nitidez média: {result['avg_blur']:.2f}\n"
            f"Tremores médios: {result['avg_jitter']:.2f}\n"
            f"Média de frequência (FFT): {result['avg_fft']:.2f}\n"
            f"Média de probabilidade de IA: {result['avg_ai_score']:.2f}%\n"
            f"Detecção por modelo:\n{per_model_info}\n"
            f"Palavras-chave suspeitas nos metadados: {', '.join(result['found_keywords']) if result['found_keywords'] else 'nenhuma'}\n"
            f"Pontuação total de suspeita: {result['score']} / 7\n"
            f"Status: {'⚠️ Provavelmente gerado por IA' if result['deepfake'] else '✅ Provavelmente real'}\n\n"
            f"--- METADADOS DO VÍDEO ---\n{json.dumps(result['metadata'], indent=2)}"
        )

        self.result_box.setText(output)
        self.select_button.setEnabled(True)

    def show_error(self, message):
        self.progress_bar.setValue(0)
        self.log_box.append(f"\n❌ ERRO:\n{message}")
        self.select_button.setEnabled(True)


def run_gui(analyze_func):
    app = QApplication(sys.argv)
    window = OlhoDeThothGUI(analyze_func)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    import cv2
import numpy as np
import mediapipe as mp
import subprocess
import json
import os
import tensorflow as tf
from tensorflow.keras.applications import (
    Xception, EfficientNetB0, InceptionV3, ResNet50
)
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
import warnings
warnings.filterwarnings('ignore')

# --------------------------
# Função 1: Obter número total de frames do vídeo
# --------------------------
def get_video_frame_count(video_path):
    ffprobe_path = os.path.join("ffmpeg", "bin", "ffprobe.exe")
    cmd = [
        ffprobe_path,
        '-v', 'error',
        '-select_streams', 'v:0',
        '-count_packets',
        '-show_entries', 'stream=nb_read_packets',
        '-of', 'default=nw=1',
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        output = result.stdout.decode()
        for line in output.splitlines():
            if "nb_read_packets" in line:
                return int(line.split("=")[1])
    except Exception as e:
        print(f"[ERRO] Não foi possível ler quantidade de frames: {e}")
    return 30  # fallback


# --------------------------
# Função 2: Extrair frames (agora com todos)
# --------------------------
def extract_frames(video_path, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        count += 1

    cap.release()
    return frames


# --------------------------
# Função 3: Detectar faces e piscadas
# --------------------------
mp_face_mesh = mp.solutions.face_mesh

def detect_faces_and_micro(frames, log_callback=None):
    face_issues = 0
    blink_count = 0

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:

        for idx, frame in enumerate(frames):
            if log_callback:
                log_callback(f"[INFO] Analisando rosto - Frame {idx + 1}/{len(frames)}")

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks is None:
                face_issues += 1
            else:
                landmarks = results.multi_face_landmarks[0].landmark
                left_eye = [landmarks[i] for i in [386, 385, 380, 374, 368, 387]]
                right_eye = [landmarks[i] for i in [159, 158, 153, 145, 144, 160]]

                def eye_aspect_ratio(eye):
                    A = abs(eye[1].y - eye[5].y)
                    B = abs(eye[2].y - eye[4].y)
                    C = abs(eye[0].x - eye[3].x)
                    return (A + B) / (2.0 * C)

                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
                if ear < 0.2:
                    blink_count += 1

    return {"face_issues": face_issues, "blink_count": blink_count}


# --------------------------
# Função 4: Estimar nitidez
# --------------------------
def estimate_blurriness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm


# --------------------------
# Função 5: Jitter entre frames
# --------------------------
def detect_frame_jitter(frames, log_callback=None):
    diffs = []
    for i in range(1, len(frames)):
        if log_callback:
            log_callback(f"[INFO] Analisando tremor - Frame {i + 1}/{len(frames)}")
        diff = cv2.absdiff(frames[i], frames[i - 1])
        mean_diff = diff.mean()
        diffs.append(mean_diff)
    return diffs


# --------------------------
# Função 6: FFT (frequência espacial)
# --------------------------
def analyze_fft(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    mean_magnitude = np.mean(magnitude_spectrum)
    return mean_magnitude


# --------------------------
# Função 7: Analisar metadados
# --------------------------
def analyze_metadata(video_path, log_callback=None):
    if log_callback:
        log_callback("[INFO] Lendo metadados do arquivo...")
    ffprobe_path = os.path.join("ffmpeg", "bin", "ffprobe.exe")
    if not os.path.isfile(ffprobe_path):
        raise FileNotFoundError(f"Arquivo ffprobe não encontrado em: {ffprobe_path}")

    cmd = [
        ffprobe_path,
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        metadata = json.loads(result.stdout)
    except json.JSONDecodeError:
        if log_callback:
            log_callback("[ERRO] Não foi possível ler os metadados do vídeo.")
        metadata = {}
    return metadata


# --------------------------
# Função 8: Carregar modelo treinado
# --------------------------

def load_model(model_name, log_callback=None):
    preprocess = None
    if model_name == "Xception":
        base_model = Xception(weights='imagenet', include_top=False)
        preprocess = xception_preprocess
    elif model_name == "EfficientNet":
        base_model = EfficientNetB0(weights='imagenet', include_top=False)
        preprocess = efficientnet_preprocess
    elif model_name == "Inception":
        base_model = InceptionV3(weights='imagenet', include_top=False)
        preprocess = inception_preprocess
    elif model_name == "ResNet":
        base_model = ResNet50(weights='imagenet', include_top=False)
        preprocess = resnet_preprocess
    else:
        raise ValueError(f"Modelo desconhecido: {model_name}")

    x = GlobalAveragePooling2D()(base_model.output)
    predictions = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model, preprocess


# Classificar frame com modelo escolhido
def classify_frame_with_ai(model, frame, preprocess_func):
    img = cv2.resize(frame, (299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_func(img_array)
    prediction = model.predict(img_array, verbose=0)
    fake_prob = prediction[0][1]
    return fake_prob


# --------------------------
# Função 9: Análise individual com um modelo
# --------------------------
def analyze_with_single_model(video_path, log_callback=None, model_type="Xception"):
    frames = extract_frames(video_path)

    model, preprocess = load_model(model_type, log_callback)
    ai_scores = []
    for i, frame in enumerate(frames):
        ai_score = classify_frame_with_ai(model, frame, preprocess)
        ai_scores.append(ai_score)
        if log_callback and i % 5 == 0:
            log_callback(f"[INFO] [{model_type}] Processando IA - Frame {i + 1}/{len(frames)}")
    avg_ai_score = np.mean(ai_scores)

    return {
        "model": model_type,
        "avg_ai_score": avg_ai_score
    }


# --------------------------
# Função 10: Análise completa com todos os modelos
# --------------------------
def analyze_with_all_models(video_path, log_callback=None):
    all_results = []

    models = ["Xception", "EfficientNet", "Inception", "ResNet"]

    # Primeiro extrair frames apenas uma vez
    if log_callback:
        log_callback("[INFO] Extraindo frames do vídeo...")

    frames = extract_frames(video_path)

    if log_callback:
        log_callback(f"[INFO] Extraídos {len(frames)} frames.")

    # Deteção de rostos (uma vez só)
    if log_callback:
        log_callback("[INFO] Analisando rostos e microcomportamentos...")
    face_data = detect_faces_and_micro(frames)

    # Nitidez média (uma vez só)
    blur_scores = [estimate_blurriness(frame) for frame in frames]
    avg_blur = np.mean(blur_scores)

    # Tremores (uma vez só)
    jitter_scores = detect_frame_jitter(frames)
    avg_jitter = np.mean(jitter_scores) if jitter_scores else 0

    # Frequências (uma vez só)
    fft_scores = [analyze_fft(frame) for frame in frames]
    avg_fft = np.mean(fft_scores)

    # Metadados (uma vez só)
    metadata = analyze_metadata(video_path, log_callback)

    # Agora analisa com todos os modelos
    ai_scores = []
    for model_type in models:
        result = analyze_with_single_model(video_path, log_callback, model_type)
        ai_scores.append(result["avg_ai_score"])
        all_results.append(result)

    avg_ai_score = np.mean(ai_scores)

    # Decisão final com base na média dos modelos
    score = 0
    if face_data['face_issues'] > 5:
        score += 2
    if face_data['blink_count'] < 2:
        score += 1
    if avg_blur < 100:
        score += 1
    if avg_jitter > 10:
        score += 1
    if avg_fft > 200:
        score += 1
    if avg_ai_score > 0.6:
        score += 2

    meta_text = json.dumps(metadata).lower()
    ai_keywords = ["google", "lavf", "ai", "synthetic", "fake"]
    found_keywords = [kw for kw in ai_keywords if kw in meta_text]

    if found_keywords:
        score += 2

    result = {
        "face_issues": face_data['face_issues'],
        "blink_count": face_data['blink_count'],
        "avg_blur": avg_blur,
        "avg_jitter": avg_jitter,
        "avg_fft": avg_fft,
        "avg_ai_score": avg_ai_score * 100,
        "score": score,
        "found_keywords": found_keywords,
        "deepfake": score >= 4,
        "metadata": metadata,
        "per_model": all_results
    }

    return result


# --------------------------
# INTERFACE GRÁFICA - PyQt5
# --------------------------

import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel,
    QTextEdit, QVBoxLayout, QWidget, QFileDialog, QMessageBox,
    QHBoxLayout,QProgressBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QTextCharFormat, QColor, QSyntaxHighlighter, QIcon, QTextCursor

class JsonHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._brushes = {
            "key": QTextCharFormat(),
            "string": QTextCharFormat(),
            "number": QTextCharFormat(),
            "boolean": QTextCharFormat(),
            "bracket": QTextCharFormat(),
        }
        self._brushes["key"].setForeground(QColor("#FFD700"))
        self._brushes["string"].setForeground(QColor("#00FF00"))
        self._brushes["number"].setForeground(QColor("#1E90FF"))
        self._brushes["boolean"].setForeground(QColor("#FF6347"))
        self._brushes["bracket"].setForeground(QColor("#FFFFFF"))

    def highlightBlock(self, text):
        import re
        patterns = [
            (r'"[^"]+":', "key"),
            (r'"[^"]*"', "string"),
            (r'\b\d+(\.\d+)?\b', "number"),
            (r'\b(true|false|null)\b', "boolean"),
            (r'[\{\}\[\]]', "bracket"),
        ]
        for pattern, key in patterns:
            for match in re.finditer(pattern, text):
                start, end = match.span()
                self.setFormat(start, end - start, self._brushes[key])


class AnalysisThread(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, video_path, analyze_func):
        super().__init__()
        self.video_path = video_path
        self.analyze_func = analyze_func

    def run(self):
        try:
            result = self.analyze_func(self.video_path, log_callback=lambda msg: self.log.emit(msg))
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class OlhoDeThothGUI(QMainWindow):
    def __init__(self, analyze_func):
        super().__init__()
        self.setWindowTitle("👁️‍🗨️ Olho de Thoth — Detector de Deepfakes")
        self.setGeometry(100, 100, 1000, 700)
        self.setStyleSheet("""
            background-color: #1e1e2f;
            color: #d9dcd6;
            font-family: 'Arial';
        """)
        self.analyze_func = analyze_func
        self.analysis_thread = None
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()

        title = QLabel("👁️‍🗨️ OLHO DE THOTH")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 32px; font-weight: bold; color: #d291bc;")
        layout.addWidget(title)

        subtitle = QLabel("Detector Avançado de Deepfakes")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("font-size: 18px; color: #aaaaaa;")
        layout.addWidget(subtitle)

        btn_layout = QHBoxLayout()
        self.select_button = QPushButton("📂 Selecionar Vídeo")
        self.select_button.clicked.connect(self.select_video)
        self.select_button.setStyleSheet(self.button_style())
        btn_layout.addWidget(self.select_button)

        self.analyze_button = QPushButton("🔍 Iniciar Análise")
        self.analyze_button.clicked.connect(self.start_analysis)
        self.analyze_button.setEnabled(False)
        self.analyze_button.setStyleSheet(self.button_style())
        btn_layout.addWidget(self.analyze_button)

        layout.addLayout(btn_layout)

        self.path_label = QLabel("Nenhum vídeo selecionado")
        self.path_label.setStyleSheet("color: #bbbbbb; margin-top: 5px;")
        layout.addWidget(self.path_label)

        # Barra de progresso
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #444;
                border-radius: 5px;
                text-align: center;
                background-color: #2a2a3d;
                color: white;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #d291bc;
                width: 20px;
            }
        """)
        layout.addWidget(self.progress_bar)

        # Logs em tempo real
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("""
            background-color: #2a2a3d;
            color: #e0e0e0;
            padding: 10px;
            font-family: Consolas;
            font-size: 12px;
            border: 1px solid #444;
        """)
        layout.addWidget(self.log_box)

        # Resultado final
        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        self.result_box.setStyleSheet("""
            background-color: #2a2a3d;
            color: #e0e0e0;
            padding: 10px;
            font-family: Consolas;
            font-size: 12px;
            border: 1px solid #444;
        """)
        self.highlighter = JsonHighlighter(self.result_box.document())
        layout.addWidget(self.result_box)
        self.result_box.hide()

        main_widget.setLayout(layout)

    def button_style(self):
        return """
            QPushButton {
                background-color: #5a2a7c;
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
                border: 1px solid #9a55ff;
            }
            QPushButton:hover {
                background-color: #7a3a9f;
            }
        """

    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Selecionar Vídeo",
            "", "Vídeos (*.mp4 *.avi *.mov *.mkv)"
        )
        if path:
            self.video_path = path
            self.path_label.setText(path)
            self.analyze_button.setEnabled(True)

    def start_analysis(self):
        if hasattr(self, 'analysis_thread') and self.analysis_thread is not None:
            try:
                self.analysis_thread.terminate()
            finally:
                self.analysis_thread = None

        self.log_box.clear()
        self.result_box.clear()
        self.result_box.hide()
        self.progress_bar.setValue(0)
        self.select_button.setEnabled(False)
        self.analyze_button.setEnabled(False)

        self.analysis_thread = AnalysisThread(self.video_path, self.analyze_func)
        self.analysis_thread.finished.connect(self.show_result)
        self.analysis_thread.error.connect(self.show_error)
        self.analysis_thread.log.connect(self.update_log)
        self.analysis_thread.start()

    def update_log(self, message):
        cursor = self.log_box.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(message + "\n")
        self.log_box.setTextCursor(cursor)
        self.log_box.ensureCursorVisible()

        # Atualiza barra de progresso com base nos frames
        if "Processando IA" in message:
            try:
                frame_num = int(message.split("Frame ")[1].split("/")[0])
                total_frames = int(message.split("/")[1].split(")")[0])
                progress = int((frame_num / total_frames) * 100)
                self.progress_bar.setValue(progress)
            except Exception:
                pass

    def show_result(self, result):
        self.progress_bar.setValue(100)
        self.log_box.hide()
        self.result_box.show()

        per_model_info = "\n".join([
            f" - {r['model']}: {r['avg_ai_score'] * 100:.2f}%"
            for r in result.get("per_model", [])
        ])

        output = (
            f"👁️‍🗨️ RESULTADO DA ANÁLISE CONSOLIDADA\n"
            f"{'-'*40}\n"
            f"Problemas faciais detectados: {result['face_issues']}\n"
            f"Piscadelas detectadas: {result['blink_count']}\n"
            f"Nitidez média: {result['avg_blur']:.2f}\n"
            f"Tremores médios: {result['avg_jitter']:.2f}\n"
            f"Média de frequência (FFT): {result['avg_fft']:.2f}\n"
            f"Média de probabilidade de IA: {result['avg_ai_score']:.2f}%\n"
            f"Detecção por modelo:\n{per_model_info}\n"
            f"Palavras-chave suspeitas nos metadados: {', '.join(result['found_keywords']) if result['found_keywords'] else 'nenhuma'}\n"
            f"Pontuação total de suspeita: {result['score']} / 7\n"
            f"Status: {'⚠️ Provavelmente gerado por IA' if result['deepfake'] else '✅ Provavelmente real'}\n\n"
            f"--- METADADOS DO VÍDEO ---\n{json.dumps(result['metadata'], indent=2)}"
        )

        self.result_box.setText(output)
        self.select_button.setEnabled(True)
        self.analyze_button.setEnabled(True)

    def show_error(self, message):
        self.progress_bar.setValue(0)
        self.log_box.append(f"\n❌ ERRO:\n{message}")
        self.select_button.setEnabled(True)
        self.analyze_button.setEnabled(True)


def run_gui(analyze_func):
    app = QApplication(sys.argv)
    window = OlhoDeThothGUI(analyze_func)
    window.show()
    sys.exit(app.exec_())

def analyze_video(video_path, log_callback=None):
    return analyze_with_all_models(video_path, log_callback)

if __name__ == "__main__":
    run_gui(analyze_video)
