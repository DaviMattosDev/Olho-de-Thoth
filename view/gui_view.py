from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel,QTextEdit, QVBoxLayout, QWidget, QFileDialog, QHBoxLayout, QProgressBar)
from PyQt5.QtGui import QColor, QTextCharFormat, QSyntaxHighlighter
from PyQt5.QtCore import Qt


# --------------------------
# Classe: JsonHighlighter
# Descrição: Destaca cores diferentes em blocos de texto JSON
# Útil para exibir metadados com sintaxe colorida
# --------------------------
class JsonHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._brushes = {
            "key": QTextCharFormat(),     # Chaves (ex: "width")
            "string": QTextCharFormat(),   # Strings (ex: "mp4")
            "number": QTextCharFormat(),   # Números (ex: 1280)
            "boolean": QTextCharFormat(),  # Booleanos (true/false)
            "bracket": QTextCharFormat()   # Colchetes e chaves ({[ ]})
        }

        # Define as cores para cada tipo de dado
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
            (r'\b\d+(\.\d+)?\b', "number"), # Número: 123 ou 12.3
            (r'\b(true|false|null)\b', "boolean"), # Valores booleanos
            (r'[\{\}\[\]]', "bracket")    # Símbolos { } [ ]
        ]

        for pattern, key in patterns:
            for match in re.finditer(pattern, text):
                start, end = match.span()
                self.setFormat(start, end - start, self._brushes[key])


# --------------------------
# Classe: OlhoDeThothGUI
# Descrição: Interface principal com PyQt5
# Contém todos os elementos visuais e estilos da GUI
# --------------------------
class OlhoDeThothGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        # Configurações gerais da janela
        self.setWindowTitle("👁️‍🗨️ Olho de Thoth — Detector de Deepfakes")
        self.setGeometry(100, 100, 1000, 700)
        self.setStyleSheet("""
            background-color: #1e1e2f;
            color: #d9dcd6;
            font-family: 'Arial';
        """)
        self.init_ui()

    # --------------------------
    # Método: init_ui
    # Descrição: Inicializa todos os componentes da interface
    # --------------------------
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

        # Layout dos botões
        btn_layout = QHBoxLayout()
        self.select_button = QPushButton("📂 Selecionar Vídeo")
        self.analyze_button = QPushButton("🔍 Iniciar Análise")

        # Estilo dos botões
        for btn in [self.select_button, self.analyze_button]:
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
        layout.addLayout(btn_layout)

        # Label do caminho do vídeo selecionado
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

        # Caixa de logs (terminal simulado)
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

        # Caixa de resultado final (em JSON)
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
        self.result_box.hide()  # Começa oculta

        # Define widget central com o layout criado
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

    # --------------------------
    # Método: clear_log
    # Descrição: Limpa a caixa de log
    # --------------------------
    def clear_log(self):
        self.log_box.clear()

    # --------------------------
    # Método: append_log
    # Descrição: Adiciona uma linha ao log
    # --------------------------
    def append_log(self, message):
        self.log_box.append(message)

    # --------------------------
    # Método: update_result
    # Descrição: Atualiza a caixa de resultados finais
    # --------------------------
    def update_result(self, result_str):
        self.result_box.setPlainText(result_str)
        self.result_box.show()