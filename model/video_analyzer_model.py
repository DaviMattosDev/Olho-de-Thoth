import cv2
import numpy as np
import mediapipe as mp
import subprocess
import json
import os
import tensorflow as tf
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
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Pega quantos frames tem o vídeo
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
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = result.stdout.decode()
        for line in output.splitlines():
            if "nb_read_packets" in line:
                return int(line.split("=")[1])
    except Exception as e:
        print(f"[ERRO] Não foi possível ler quantidade de frames: {e}")
    return 30  # se der ruim, retorna 30

# Extrai todos os frames do vídeo
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

# Detecta rosto e piscadelas
mp_face_mesh = mp.solutions.face_mesh
def detect_faces_and_micro(frames, log_callback=None):
    face_issues = 0
    blink_count = 0

    def ear(olho):
        A = abs(olho[1].y - olho[5].y)
        B = abs(olho[2].y - olho[4].y)
        C = abs(olho[0].x - olho[3].x)
        return (A + B) / (2 * C)

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        for idx, frame in enumerate(frames):
            if log_callback:
                log_callback(f"[INFO] Analisando rosto - Frame {idx + 1}/{len(frames)}")

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks is None:
                face_issues += 1
            else:
                lms = results.multi_face_landmarks[0].landmark
                left_eye = [lms[i] for i in [386, 385, 380, 374, 368, 387]]
                right_eye = [lms[i] for i in [159, 158, 153, 145, 144, 160]]

                esq = ear(left_eye)
                dir = ear(right_eye)
                media = (esq + dir) / 2

                if media < 0.2:
                    blink_count += 1

    return {"face_issues": face_issues, "blink_count": blink_count}

# Calcula nitidez do frame
def estimate_blurriness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# Verifica tremor entre frames
def detect_frame_jitter(frames, log_callback=None):
    diffs = []
    for i in range(1, len(frames)):
        if log_callback:
            log_callback(f"[INFO] Analisando tremor - Frame {i + 1}/{len(frames)}")
        diff = cv2.absdiff(frames[i], frames[i - 1])
        diffs.append(diff.mean())
    return diffs

# Análise de frequência espacial com FFT
def analyze_fft(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fshift = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = 20 * np.log(np.abs(fshift) + 1e-6)
    return np.std(magnitude)

# Lê metadados do vídeo
def analyze_metadata(video_path, log_callback=None):
    ffprobe_path = os.path.join("ffmpeg", "bin", "ffprobe.exe")
    if not os.path.isfile(ffprobe_path):
        raise FileNotFoundError(f"Não achei o ffprobe: {ffprobe_path}")

    if log_callback:
        log_callback("[INFO] Lendo metadados...")

    cmd = [
        ffprobe_path,
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        video_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE)
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        if log_callback:
            log_callback("[ERRO] Não consegui ler metadados.")
        return {}

# Carrega modelo específico
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
    base = base_model(weights='imagenet', include_top=False)
    x = GlobalAveragePooling2D()(base.output)
    output = Dense(2, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=output)
    return model, preprocess

# Classifica frame com ROI facial
def classify_frame_with_ai(model, frame, preprocess_func):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        h, w, _ = frame.shape
        lms = results.multi_face_landmarks[0].landmark
        x_min = min(int(l.x * w) for l in lms)
        x_max = max(int(l.x * w) for l in lms)
        y_min = min(int(l.y * h) for l in lms)
        y_max = max(int(l.y * h) for l in lms)
        roi = frame[y_min:y_max, x_min:x_max]
        img = cv2.resize(roi, (299, 299))
        arr = image.img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_func(arr)
        prediction = model.predict(arr, verbose=0)
        return prediction[0][1]
    return None

# Analisa com só um modelo (pra não dar pau na memória)
def analyze_with_single_model(video_path, model_type="Xception", log_callback=None, total_frames=0, progress_callback=None):
    model, preprocess = load_model(model_type)
    ai_scores = []
    frames = extract_frames(video_path)
    for i, frame in enumerate(frames):
        score = classify_frame_with_ai(model, frame, preprocess)
        if score is not None:
            ai_scores.append(score)
        if log_callback and i % 5 == 0:
            log_callback(f"[INFO] [{model_type}] Processando IA - Frame {i + 1}/{len(frames)}")
        if progress_callback:
            progress_callback(int((i + 1) / total_frames * 100))
    avg_score = np.mean(ai_scores) if ai_scores else 0
    K.clear_session()
    return {"model": model_type, "avg_ai_score": avg_score}

# Detecção de respiração natural (movimento mandibular/pescoço)
def detect_natural_breathing(frames, log_callback=None):
    neck_movement = []
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
        for idx, frame in enumerate(frames):
            if log_callback:
                log_callback(f"[INFO] Analisando respiração - Frame {idx + 1}/{len(frames)}")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                lms = results.multi_face_landmarks[0].landmark
                # Pontos do queixo e pescoço
                chin_y = lms[152].y
                neck_y = lms[12].y  # Exemplo aproximado
                movement = abs(chin_y - neck_y)
                neck_movement.append(movement)
    if len(neck_movement) < 10:
        return False
    peaks, _ = find_peaks(neck_movement, height=np.mean(neck_movement)+0.01, distance=10)
    return len(peaks) >= 3

# OCR para detectar marcas d'água invisíveis
def detect_watermark_ocr(frames, log_callback=None):
    detected_text = set()
    for idx, frame in enumerate(frames):
        if log_callback and idx % 10 == 0:
            log_callback(f"[INFO] Buscando marca d'água - Frame {idx + 1}/{len(frames)}")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(thresh, config='--psm 6')
        if text.strip():
            detected_text.add(text.strip())
    return list(detected_text)

# Função principal de análise
def analyze_video(video_path, log_callback=None, progress_callback=None, model_type="Xception"):
    frames = extract_frames(video_path)
    if log_callback:
        log_callback(f"[INFO] Frames extraídos: {len(frames)}")

    # Análise facial
    face_data = detect_faces_and_micro(frames, log_callback)

    # Nitidez
    blur_scores = [estimate_blurriness(f) for f in frames]
    avg_blur = np.mean(blur_scores)

    # Tremores
    jitter_scores = detect_frame_jitter(frames, log_callback)
    avg_jitter = np.mean(jitter_scores) if jitter_scores else 0

    # Frequências
    fft_scores = [analyze_fft(f) for f in frames]
    avg_fft = np.mean(fft_scores)

    # Respiração
    has_natural_breathing = detect_natural_breathing(frames, log_callback)

    # OCR
    watermark_texts = detect_watermark_ocr(frames, log_callback)

    # Metadados
    metadata = analyze_metadata(video_path, log_callback)

    # Usa só um modelo pra não lotar RAM
    result = analyze_with_single_model(
        video_path,
        model_type=model_type,
        log_callback=log_callback,
        total_frames=len(frames),
        progress_callback=progress_callback
    )
    avg_ai_score = result["avg_ai_score"]

    # Pontuação final
    score = 0
    if face_data['face_issues'] > 5:
        score += 1
    if face_data['blink_count'] < 2:
        score += 0.5
    if avg_blur < 100:
        score += 0.5
    if avg_jitter > 10:
        score += 0.5
    if avg_fft > 200:
        score += 0.5
    if avg_ai_score > 0.6:
        score += 1.5
    if not has_natural_breathing:
        score += 1
    if watermark_texts:
        score += 0.5

    meta_text = json.dumps(metadata).lower()
    ai_keywords = [
        "deepfake", "synthetic", "fake", "ai", "generated", "gan", "neural",
        "stylegan", "ffmpeg", "lavf", "google", "ai", "fake", "gerado"
    ]
    found_keywords = [kw for kw in ai_keywords if kw in meta_text]
    if found_keywords:
        score += 1

    return {
        "face_issues": face_data['face_issues'],
        "blink_count": face_data['blink_count'],
        "avg_blur": avg_blur,
        "avg_jitter": avg_jitter,
        "avg_fft": avg_fft,
        "has_natural_breathing": has_natural_breathing,
        "watermark_texts": watermark_texts,
        "avg_ai_score": avg_ai_score * 100,
        "score": score,
        "found_keywords": found_keywords,
        "deepfake": score >= 3.5 or avg_ai_score > 0.7,
        "metadata": metadata,
        "per_model": [result]
    }
