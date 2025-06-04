import json
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QFileDialog
from view.gui_view import JsonHighlighter
from model.video_analyzer_model import analyze_video
from model.info_model import get_model_description


# --------------------------
# Classe: AnalysisWorker
# Descrição: Thread secundária para rodar análise sem travar a interface
# Signals:
#   - finished: Resultado final da análise (dict)
#   - error: Emite uma mensagem de erro (str)
#   - log: Emite mensagens para o log da interface (str)
#   - progress: Atualiza a barra de progresso (int 0-100)
# --------------------------
class AnalysisWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    log = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, video_path, model="Xception"):
        super().__init__()
        self.video_path = video_path  # Caminho do vídeo a ser analisado
        self.model = model            # Modelo de IA selecionado pelo usuário

    def run(self):
        try:
            # Executa a análise do vídeo em uma thread separada
            result = analyze_video(
                self.video_path,
                log_callback=self.log.emit,         # Para atualizar o log em tempo real
                progress_callback=self.progress.emit, # Para atualizar a barra de progresso
                model_type=self.model               # Usando apenas o modelo selecionado
            )
            self.finished.emit(result)  # Envia resultado final para a GUI
        except Exception as e:
            self.error.emit(str(e))    # Caso ocorra erro, envia a mensagem de erro


# --------------------------
# Classe: AppController
# Descrição: Controlador principal do sistema (MVC - Controller)
# Conecta os eventos da interface com as funções do modelo
# --------------------------
class AppController:
    def __init__(self, view):
        self.view = view
        self.worker = None

        # Conecta os eventos da interface (botões, combo box etc.)
        self.view.select_button.clicked.connect(self.select_video)
        self.view.analyze_button.clicked.connect(self.start_analysis)
        self.view.model_combo.currentIndexChanged.connect(self.update_model_description)

        # Carrega descrição inicial do modelo na inicialização
        self.update_model_description()

    # --------------------------
    # Método: update_model_description
    # Descrição: Atualiza o log com a descrição do modelo selecionado
    # --------------------------
    def update_model_description(self):
        model = self.view.model_combo.currentText()
        description = get_model_description(model)
        self.view.clear_log()
        self.view.append_log(description)

    # --------------------------
    # Método: select_video
    # Descrição: Abre um diálogo para selecionar o vídeo
    # Atualiza o label e habilita o botão de análise
    # --------------------------
    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self.view, "Selecionar Vídeo",
            "", "Vídeos (*.mp4 *.avi *.mov *.mkv)"
        )
        if path:
            self.video_path = path
            self.view.path_label.setText(path)
            self.view.analyze_button.setEnabled(True)
            self.update_model_description()  # Mantém a descrição visível

    # --------------------------
    # Método: start_analysis
    # Descrição: Inicia a análise em uma thread separada
    # --------------------------
    def start_analysis(self):
        self.view.clear_log()
        self.view.result_box.hide()
        self.view.progress_bar.setValue(0)
        self.view.select_button.setEnabled(False)
        self.view.analyze_button.setEnabled(False)

        # Pega o modelo selecionado no combobox
        model = self.view.model_combo.currentText()

        # Cria e inicia o worker (thread de análise)
        self.worker = AnalysisWorker(self.video_path, model=model)
        self.worker.finished.connect(self.handle_analysis_finished)
        self.worker.error.connect(self.handle_error)
        self.worker.log.connect(self.view.append_log)
        self.worker.progress.connect(self.view.progress_bar.setValue)
        self.worker.start()

    # --------------------------
    # Método: handle_analysis_finished
    # Descrição: Processa o resultado final da análise e mostra na GUI
    # --------------------------
    def handle_analysis_finished(self, result):
        # Formata informações por modelo
        per_model_info = "\n".join([
            f" - {r['model']}: {r['avg_ai_score']:.2f}%"
            for r in result.get("per_model", [])
        ])

        # Monta o texto do relatório final
        output = (
            f"👁️‍🗨️ RESULTADO DA ANÁLISE\n"
            f"{'-' * 40}\n"
            f"Problemas faciais detectados: {result['face_issues']}\n"
            f"Piscadelas detectadas: {result['blink_count']}\n"
            f"Nitidez média: {result['avg_blur']:.2f}\n"
            f"Tremores médios: {result['avg_jitter']:.2f}\n"
            f"Média de frequência (FFT): {result['avg_fft']:.2f}\n"
            f"Probabilidade média de IA: {result['avg_ai_score']:.2f}%\n"
            f"Detecção por modelo:\n{per_model_info}\n"
            f"Palavras-chave suspeitas: {', '.join(result['found_keywords']) if result['found_keywords'] else 'nenhuma'}\n"
            f"Pontuação final: {result['score']} / 7\n"
            f"Status: {'⚠️ Provavelmente gerado por IA' if result['deepfake'] else '✅ Provavelmente real'}\n"
            f"--- METADADOS ---\n{json.dumps(result['metadata'], indent=2)}"
        )

        # Atualiza a caixa de resultados
        self.view.update_result(output)

        # Completa a barra de progresso
        self.view.progress_bar.setValue(100)

        # Reativa o botão de seleção de vídeo
        self.view.select_button.setEnabled(True)

    # --------------------------
    # Método: handle_error
    # Descrição: Trata erros durante a análise e exibe no log
    # --------------------------
    def handle_error(self, error_msg):
        self.view.append_log(f"\n❌ ERRO:\n{error_msg}")
        self.view.select_button.setEnabled(True)