from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QTextEdit,
    QVBoxLayout, QWidget, QFileDialog, QHBoxLayout, QProgressBar, QComboBox
)
from PyQt5.QtGui import QColor, QTextCharFormat, QSyntaxHighlighter
from PyQt5.QtCore import Qt


# --------------------------
# Classe: JsonHighlighter
# Destaque de sintaxe para JSON
# --------------------------
class JsonHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._brushes = {
            "key": QTextCharFormat(),
            "string": QTextCharFormat(),
            "number": QTextCharFormat(),
            "boolean": QTextCharFormat(),
            "bracket": QTextCharFormat()
        }

        # Cores para cada tipo de dado
        self._brushes["key"].setForeground(QColor("#FFD700"))     # Amarelo dourado
        self._brushes["string"].setForeground(QColor("#00FF00")) # Verde
        self._brushes["number"].setForeground(QColor("#1E90FF")) # Azul
        self._brushes["boolean"].setForeground(QColor("#FF6347")) # Vermelho claro
        self._brushes["bracket"].setForeground(QColor("#FFFFFF")) # Branco

    def highlightBlock(self, text):
        import re
        patterns = [
            (r'"[^"]+":', "key"),         # Chave: "key":
            (r'"[^"]*"', "string"),       # String: "valor"
            (r'\b\d+(\.\d+)?\b', "number"), # Número
            (r'\b(true|false|null)\b', "boolean"), # Booleanos
            (r'[\{\}\[\]]', "bracket")    # Símbolos { } [ ]
        ]

        for pattern, key in patterns:
            for match in re.finditer(pattern, text):
                start, end = match.span()
                self.setFormat(start, end - start, self._brushes[key])


# --------------------------
# Classe: OlhoDeThothGUI
# Interface principal da aplicação
# --------------------------
class OlhoDeThothGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("👁️‍🗨️ Olho de Thoth — Detector de Deepfakes")
        self.setGeometry(100, 100, 1000, 700)
        self.setStyleSheet("""
            background-color: #1e1e2f;
            color: #d9dcd6;
            font-family: 'Arial';
        """)
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        layout = QVBoxLayout()

        # Título principal
        title = QLabel("👁️‍🗨️ OLHO DE THOTH")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 32px; font-weight: bold; color: #d291bc;")
        layout.addWidget(title)

        # Subtítulo
        subtitle = QLabel("Detector Avançado de Deepfakes")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("font-size: 18px; color: #aaaaaa;")
        layout.addWidget(subtitle)

        # Seção de seleção de modelo
        model_title = QLabel("🧠 Escolha o Modelo de Inteligência Artificial")
        model_title.setAlignment(Qt.AlignCenter)
        model_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #d9dcd6; margin-top: 20px;")
        layout.addWidget(model_title)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["Xception", "EfficientNet", "Inception", "ResNet"])
        self.model_combo.setCurrentText("Xception")
        self.model_combo.setFixedWidth(250)
        self.model_combo.setStyleSheet("""
            QComboBox {
                background-color: #2a2a3d;
                color: #e0e0e0;
                padding: 8px;
                font-size: 14px;
                border-radius: 5px;
                border: 1px solid #555;
            }
            QComboBox:hover {
                border: 1px solid #d291bc;
            }
        """)

        combo_layout = QHBoxLayout()
        combo_layout.addStretch()
        combo_layout.addWidget(self.model_combo)
        combo_layout.addStretch()
        layout.addLayout(combo_layout)

        # Botões principais
        btn_layout = QHBoxLayout()
        self.select_button = QPushButton("📂 Selecionar Vídeo")
        self.analyze_button = QPushButton("🔍 Iniciar Análise")
        self.clear_button = QPushButton("🧹 Limpar Log e Análise")

        for btn in [self.select_button, self.analyze_button, self.clear_button]:
            btn.setStyleSheet("""
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
            """)

        btn_layout.addWidget(self.select_button)
        btn_layout.addWidget(self.analyze_button)
        btn_layout.addWidget(self.clear_button)
        layout.addLayout(btn_layout)

        # Caminho do vídeo selecionado
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
            }
        """)
        layout.addWidget(self.progress_bar)

        # Log de análise
        log_title = QLabel("🧾 Log de Análise:")
        log_title.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 10px;")
        layout.addWidget(log_title)

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

        # Resultado em JSON
        result_title = QLabel("📊 Resultado JSON:")
        result_title.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 10px;")
        layout.addWidget(result_title)

        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        self.result_box.setMinimumHeight(200)
        self.result_box.setStyleSheet("""
            background-color: #2a2a3d;
            color: #e0e0e0;
            padding: 10px;
            font-family: Consolas;
            font-size: 12px;
            border: 1px solid #444;
        """)
        self.highlighter = JsonHighlighter(self.result_box.document())
        self.result_box.hide()
        layout.addWidget(self.result_box)

        # Define widget central
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

        # Exibe a descrição do modelo ao iniciar
        self.update_model_description()

    def update_model_description(self):
        from model.info_model import get_model_description
        model = self.model_combo.currentText()
        description = get_model_description(model)
        self.clear_log()
        self.append_log(description)

    # Limpa o log e o resultado JSON
    def clear_analysis(self):
        self.clear_log()
        self.result_box.hide()

    # Limpa o log
    def clear_log(self):
        self.log_box.clear()

    # Adiciona uma linha ao log
    def append_log(self, message):
        self.log_box.append(message)

    # Mostra resultado final em formato JSON
    def update_result(self, result_str):
        self.result_box.setPlainText(result_str)
        self.result_box.show()