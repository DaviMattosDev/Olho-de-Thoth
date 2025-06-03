# gui.py

import sys
import json
import re
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel,
    QTextEdit, QVBoxLayout, QWidget, QFileDialog, QHBoxLayout,
    QProgressBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QTextCharFormat, QColor, QSyntaxHighlighter, QTextCursor


# Destaque de sintaxe para JSON
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


# Thread para análise em segundo plano
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


# Janela principal da GUI
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
            f" - {r['model']}: {r['avg_ai_score']:.2f}%"
            for r in result.get("per_model", [])
        ])

        output = (
            f"👁️‍🗨️ RESULTADO DA ANÁLISE CONSOLIDADA\n"
            f"{'-' * 40}\n"
            f"Problemas faciais detectados: {result['face_issues']}\n"
            f"Piscadelas detectadas: {result['blink_count']}\n"
            f"Nitidez média: {result['avg_blur']:.2f}\n"
            f"Tremores médios: {result['avg_jitter']:.2f}\n"
            f"Média de frequência (FFT): {result['avg_fft']:.2f}\n"
            f"Média de probabilidade de IA: {result['avg_ai_score']:.2f}%\n"
            f"Detecção por modelo:\n{per_model_info}\n"
            f"Palavras-chave suspeitas nos metadados: {', '.join(result['found_keywords']) if result['found_keywords'] else 'nenhuma'}\n"
            f"Pontuação total de suspeita: {result['score']} / 7\n"
            f"Status: {'⚠️ Provavelmente gerado por IA' if result['deepfake'] else '✅ Provavelmente real'}\n"
            f"--- METADADOS DO VÍDEO ---\n"
            f"{json.dumps(result['metadata'], indent=2)}"
        )

        self.result_box.setText(output)
        self.select_button.setEnabled(True)
        self.analyze_button.setEnabled(True)

    def show_error(self, message):
        self.progress_bar.setValue(0)
        self.log_box.append(f"\n❌ ERRO:\n{message}")
        self.select_button.setEnabled(True)
        self.analyze_button.setEnabled(True)


# Função principal para rodar a GUI
def run_gui(analyze_func):
    app = QApplication(sys.argv)
    window = OlhoDeThothGUI(analyze_func)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    from analyse import analyze_video
    run_gui(analyze_video)