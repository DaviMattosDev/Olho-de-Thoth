import cv2
import numpy as np
import mediapipe as mp
import subprocess
import json
import os
import tensorflow as tf
import ffmpeg
from tensorflow.keras.applications import Xception, EfficientNetB0, InceptionV3, ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import warnings
import pytesseract
from scipy.signal import find_peaks

warnings.filterwarnings('ignore')
mp_face_mesh = mp.solutions.face_mesh

def load_config():
    """Carrega as configurações do arquivo config.json com valores padrão de fallback."""
    defaults = {
        "scoring": {
            "deepfake_threshold": 3.5,
            "ai_score_threshold": 0.7,
            "weights": {
                "face_issues": 1.0, "blink_count": 0.5, "blurriness": 0.5,
                "jitter": 0.5, "fft": 0.5, "ai_model": 1.5,
                "natural_breathing": 1.0, "watermark": 0.5, "metadata_keywords": 1.0
            }
        },
        "thresholds": {
            "face_issues_max": 5, "blink_count_min": 2, "blurriness_max": 100.0,
            "jitter_min": 10.0, "fft_min": 200.0, "ai_model_min": 0.6
        },
        "metadata": {
            "ai_keywords": [
                "deepfake", "synthetic", "fake", "ai", "generated", "gan",
                "neural", "stylegan", "ffmpeg", "lavf", "google", "gerado"
            ]
        }
    }
    try:
        with open("config.json", "r") as f:
            user_config = json.load(f)
            # A simple merge could be implemented here if necessary
            return user_config
    except (FileNotFoundError, json.JSONDecodeError):
        return defaults

# Pega quantos frames tem o vídeo
def get_video_frame_count(video_path):
    """Conta os frames de um vídeo usando ffprobe."""
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        if 'nb_frames' in video_stream and int(video_stream['nb_frames']) > 0:
            return int(video_stream['nb_frames'])
        if 'nb_read_packets' in video_stream:
            return int(video_stream['nb_read_packets'])
        return 30 # Fallback
    except (ffmpeg.Error, StopIteration, KeyError) as e:
        print(f"[ERRO] Não foi possível ler a quantidade de frames com ffprobe: {e}")
        return 30  # Retorna um valor padrão em caso de erro

# Lê metadados do vídeo
def analyze_metadata(video_path, log_callback=None):
    """Lê metadados do vídeo usando ffprobe."""
    if log_callback:
        log_callback("[INFO] Lendo metadados...")
    try:
        probe = ffmpeg.probe(video_path)
        return probe
    except ffmpeg.Error as e:
        if log_callback:
            error_message = (e.stderr or b'').decode('utf-8')
            log_callback(f"[ERRO] Não consegui ler metadados: {error_message}")
        return {}


class FaceAndMicroExpressionAnalyzer:
    def __init__(self, log_callback=None):
        self.face_issues = 0
        self.blink_count = 0
        self.log_callback = log_callback
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

    def _ear(self, olho):
        A = abs(olho[1].y - olho[5].y)
        B = abs(olho[2].y - olho[4].y)
        C = abs(olho[0].x - olho[3].x)
        return (A + B) / (2 * C)

    def analyze_frame(self, frame, frame_idx, total_frames):
        if self.log_callback and frame_idx % 10 == 0:
            self.log_callback(f"[INFO] Analisando rosto - Frame {frame_idx + 1}/{total_frames}")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks is None:
            self.face_issues += 1
        else:
            lms = results.multi_face_landmarks[0].landmark
            left_eye = [lms[i] for i in [386, 385, 380, 374, 368, 387]]
            right_eye = [lms[i] for i in [159, 158, 153, 145, 144, 160]]

            esq = self._ear(left_eye)
            dir = self._ear(right_eye)
            media = (esq + dir) / 2

            if media < 0.2:
                self.blink_count += 1

    def get_results(self):
        return {"face_issues": self.face_issues, "blink_count": self.blink_count}


class BlurAnalyzer:
    def __init__(self):
        self.blur_scores = []

    def analyze_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        self.blur_scores.append(score)

    def get_results(self):
        if not self.blur_scores:
            return 0
        return np.mean(self.blur_scores)


class JitterAnalyzer:
    def __init__(self, log_callback=None):
        self.jitter_scores = []
        self.previous_frame = None
        self.log_callback = log_callback

    def analyze_frame(self, frame, frame_idx, total_frames):
        if self.previous_frame is not None:
            if self.log_callback and frame_idx % 10 == 0:
                self.log_callback(f"[INFO] Analisando tremor - Frame {frame_idx + 1}/{total_frames}")
            diff = cv2.absdiff(frame, self.previous_frame)
            self.jitter_scores.append(diff.mean())
        self.previous_frame = frame.copy()

    def get_results(self):
        if not self.jitter_scores:
            return 0
        return np.mean(self.jitter_scores)


class FftAnalyzer:
    def __init__(self):
        self.fft_scores = []

    def analyze_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fshift = np.fft.fftshift(np.fft.fft2(gray))
        magnitude = 20 * np.log(np.abs(fshift) + 1e-6)
        self.fft_scores.append(np.std(magnitude))

    def get_results(self):
        if not self.fft_scores:
            return 0
        return np.mean(self.fft_scores)


class BreathingAnalyzer:
    def __init__(self, log_callback=None):
        self.neck_movements = []
        self.log_callback = log_callback
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    def analyze_frame(self, frame, frame_idx, total_frames):
        if self.log_callback and frame_idx % 10 == 0:
            self.log_callback(f"[INFO] Analisando respiração - Frame {frame_idx + 1}/{total_frames}")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if results.multi_face_landmarks:
            lms = results.multi_face_landmarks[0].landmark
            chin_y = lms[152].y
            neck_y = lms[12].y
            movement = abs(chin_y - neck_y)
            self.neck_movements.append(movement)

    def get_results(self):
        if len(self.neck_movements) < 10:
            return False
        if not self.neck_movements:
            return False
        mean_movement = np.mean(self.neck_movements)
        peaks, _ = find_peaks(self.neck_movements, height=mean_movement + 0.01, distance=10)
        return len(peaks) >= 3


class WatermarkAnalyzer:
    def __init__(self, log_callback=None):
        self.detected_text = set()
        self.log_callback = log_callback

    def analyze_frame(self, frame, frame_idx, total_frames):
        if frame_idx % 15 == 0:
            if self.log_callback:
                self.log_callback(f"[INFO] Buscando marca d'água - Frame {frame_idx + 1}/{total_frames}")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
            try:
                text = pytesseract.image_to_string(thresh, config='--psm 6', timeout=2)
                if text.strip():
                    self.detected_text.add(text.strip())
            except pytesseract.TesseractError:
                if self.log_callback:
                    self.log_callback(f"[AVISO] Tesseract falhou ou demorou demais no frame {frame_idx + 1}")

    def get_results(self):
        return list(self.detected_text)


def load_model(model_name):
    modelos = {
        "Xception": (Xception, xception_preprocess),
        "EfficientNet": (EfficientNetB0, efficientnet_preprocess),
        "Inception": (InceptionV3, inception_preprocess),
        "ResNet": (ResNet50, resnet_preprocess)
    }

    if model_name not in modelos:
        raise ValueError(f"Modelo desconhecido: {model_name}")

    base_model, preprocess = modelos[model_name]
    base = base_model(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    x = GlobalAveragePooling2D()(base.output)
    output = Dense(2, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=output)
    return model, preprocess


class AIFrameClassifier:
    def __init__(self, model_type="Xception", log_callback=None):
        self.log_callback = log_callback
        self.model_type = model_type
        self.model, self.preprocess_func = load_model(model_type)
        self.ai_scores = []
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

    def classify_frame(self, frame, frame_idx, total_frames):
        if self.log_callback and frame_idx % 5 == 0:
            self.log_callback(f"[INFO] [{self.model_type}] Processando IA - Frame {frame_idx + 1}/{total_frames}")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return

        h, w, _ = frame.shape
        lms = results.multi_face_landmarks[0].landmark
        try:
            x_min = min(int(l.x * w) for l in lms)
            x_max = max(int(l.x * w) for l in lms)
            y_min = min(int(l.y * h) for l in lms)
            y_max = max(int(l.y * h) for l in lms)

            pad = 10
            roi = frame[max(0, y_min - pad):min(h, y_max + pad), max(0, x_min - pad):min(w, x_max + pad)]

            if roi.size == 0:
                return

            img = cv2.resize(roi, (299, 299))
            arr = image.img_to_array(img)
            arr = np.expand_dims(arr, axis=0)
            arr = self.preprocess_func(arr)
            prediction = self.model.predict(arr, verbose=0)
            self.ai_scores.append(prediction[0][1])
        except Exception as e:
            if self.log_callback:
                self.log_callback(f"[AVISO] Falha ao processar ROI da IA no frame {frame_idx + 1}: {e}")

    def get_results(self):
        K.clear_session()
        avg_score = np.mean(self.ai_scores) if self.ai_scores else 0
        return {"model": self.model_type, "avg_ai_score": avg_score}


def calculate_final_score(results, config):
    """Calcula a pontuação final com base nos resultados da análise e na configuração."""
    score = 0
    weights = config['scoring']['weights']
    thresholds = config['thresholds']

    # Regras para pontuação numérica
    rules = [
        {'metric': 'face_issues', 'op': '>', 'threshold': thresholds['face_issues_max'], 'weight': weights['face_issues']},
        {'metric': 'blink_count', 'op': '<', 'threshold': thresholds['blink_count_min'], 'weight': weights['blink_count']},
        {'metric': 'avg_blur', 'op': '<', 'threshold': thresholds['blurriness_max'], 'weight': weights['blurriness']},
        {'metric': 'avg_jitter', 'op': '>', 'threshold': thresholds['jitter_min'], 'weight': weights['jitter']},
        {'metric': 'avg_fft', 'op': '>', 'threshold': thresholds['fft_min'], 'weight': weights['fft']},
        # Note: avg_ai_score is 0-1, not 0-100 here
        {'metric': 'avg_ai_score', 'op': '>', 'threshold': thresholds['ai_model_min'], 'weight': weights['ai_model']},
    ]

    for rule in rules:
        value = results.get(rule['metric'], 0)
        if (rule['op'] == '>' and value > rule['threshold']) or \
           (rule['op'] == '<' and value < rule['threshold']):
            score += rule['weight']

    # Casos especiais (booleanos e listas)
    if not results.get('has_natural_breathing'):
        score += weights['natural_breathing']
    if results.get('watermark_texts'):
        score += weights['watermark']

    # Keywords nos metadados
    meta_text = json.dumps(results.get('metadata', {})).lower()
    ai_keywords = config['metadata']['ai_keywords']
    found_keywords = [kw for kw in ai_keywords if kw in meta_text]
    if found_keywords:
        score += weights['metadata_keywords']

    # Atualiza o dicionário de resultados
    results['found_keywords'] = list(set(found_keywords))
    results['score'] = score

    # Determinação final
    deepfake_score_threshold = config['scoring']['deepfake_threshold']
    ai_score_threshold = config['scoring']['ai_score_threshold']
    results['deepfake'] = score >= deepfake_score_threshold or results.get('avg_ai_score', 0) > ai_score_threshold

    # Formata o score de IA para exibição (0-100)
    results['avg_ai_score'] *= 100

    return results

# Função principal de análise
def analyze_video(video_path, log_callback=None, progress_callback=None, model_type="Xception"):
    if log_callback:
        log_callback(f"[INFO] Iniciando análise do vídeo: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Não foi possível abrir o arquivo de vídeo: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = get_video_frame_count(video_path)

    if log_callback:
        log_callback(f"[INFO] Total de frames detectados: {total_frames}")

    # Initialize analyzers
    face_analyzer = FaceAndMicroExpressionAnalyzer(log_callback)
    blur_analyzer = BlurAnalyzer()
    jitter_analyzer = JitterAnalyzer(log_callback)
    fft_analyzer = FftAnalyzer()
    breathing_analyzer = BreathingAnalyzer(log_callback)
    watermark_analyzer = WatermarkAnalyzer(log_callback)
    ai_classifier = AIFrameClassifier(model_type, log_callback)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run analyzers for the current frame
        face_analyzer.analyze_frame(frame, frame_idx, total_frames)
        blur_analyzer.analyze_frame(frame)
        jitter_analyzer.analyze_frame(frame, frame_idx, total_frames)
        fft_analyzer.analyze_frame(frame)
        breathing_analyzer.analyze_frame(frame, frame_idx, total_frames)
        watermark_analyzer.analyze_frame(frame, frame_idx, total_frames)
        ai_classifier.classify_frame(frame, frame_idx, total_frames)

        if progress_callback and total_frames > 0:
            progress = int((frame_idx + 1) / total_frames * 100)
            progress_callback(progress)

        frame_idx += 1

    cap.release()

    if log_callback:
        log_callback("[INFO] Análise de frames concluída. Compilando resultados...")

    # Compila todos os resultados em um único dicionário
    face_data = face_analyzer.get_results()
    ai_result = ai_classifier.get_results()

    analysis_results = {
        "face_issues": face_data['face_issues'],
        "blink_count": face_data['blink_count'],
        "avg_blur": blur_analyzer.get_results(),
        "avg_jitter": jitter_analyzer.get_results(),
        "avg_fft": fft_analyzer.get_results(),
        "has_natural_breathing": breathing_analyzer.get_results(),
        "watermark_texts": watermark_analyzer.get_results(),
        "avg_ai_score": ai_result["avg_ai_score"], # Score de 0-1
        "metadata": analyze_metadata(video_path, log_callback),
        "per_model": [ai_result]
    }

    # Carrega a configuração e calcula a pontuação final
    config = load_config()
    final_results = calculate_final_score(analysis_results, config)

    return final_results
