
# 👁️ Olho de Thoth

<div align="center">
  <img src="https://media1.tenor.com/m/DvFsvcuRZC0AAAAd/ra-amun.gif" alt="Olho-de-thoth" height='300px' width='400px'>
</div>

🧠 **"O Olho de Thoth vê além da ilusão."**  
Um sistema híbrido para detectar vídeos provavelmente gerados por inteligência artificial — usando análise visual, aprendizado de máquina e metadados.

---

## 🌟 Sobre

**Olho-de-Thoth** é uma ferramenta inspirada na sabedoria do deus egípcio Thoth, capaz de *"ver"* se um vídeo foi criado por IA ou gravado na vida real.  
Utiliza técnicas avançadas de visão computacional, modelos pré-treinados (como **XceptionNet**), análise de artefatos visuais e leitura de metadados para identificar pistas sutis de vídeos sintéticos.

Com a explosão de modelos como **Google VEO 3**, **OpenAI Sora** e **Stable Video Diffusion**, torna-se essencial ter sistemas capazes de distinguir o real do falso.  
O Olho-de-Thoth nasce dessa necessidade: ser o olho que enxerga a verdade por trás da tecnologia.

---

## 🔍 Funcionalidades

- ✅ **Análise facial:** detecta inconsistências faciais e ausência de piscadelas  
- 🤖 **Uso de modelo de IA (XceptionNet ou outro)** para classificação frame a frame  
- 📸 **Detecção de nitidez anormal**  
- 🎞️ **Análise de tremor entre frames (jitter)**  
- 📊 **Análise de frequência espacial (FFT)**  
- 🧬 **Leitura de metadados do vídeo** (via FFmpeg / ffprobe)  
- 🫁 **Detecta respiração natural** (útil contra deepfakes avançados)  
- 💧 **Verifica presença de marcas d’água** (visíveis ou discretas)  
- 🧮 **Sistema de pontuação final para decisão**  

---

## 🧪 Como funciona?

O detector analisa o vídeo em múltiplas camadas:

1. Extração de frames do vídeo  
2. Análise facial e microcomportamentos  
3. Medição de nitidez e jitter  
4. Transformada de Fourier (FFT) para padrões espaciais  
5. Classificação com modelo **XceptionNet ou outro**  
6. Leitura de metadados do arquivo  
7. Verificação de respiração e OCR para marcas d’água  
8. Atribuição de pontuação e decisão final  

> 🔍 **Se a soma das pistas for maior ou igual a 4 → Provavelmente gerado por IA**  
> 🔍 **Para vídeos muito realistas (ex: Google VEO 3), a IA pode elevar o limiar para 5+**

---
## ⚠️ ATENÇÃO ⚠️

Para garantir maior precisão na detecção de vídeos gerados por inteligência artificial, o sistema utiliza mais de um modelo de análise. Cada modelo foi escolhido por suas capacidades específicas em detectar artefatos visuais, padrões anômalos ou ausência de características humanas naturais.

Como os vídeos sintéticos estão cada vez mais realistas — e, em muitos casos, a qualidade do vídeo analisado pode estar comprometida (baixa resolução, compressão, ruído) — falsos positivos podem ocorrer.

Por isso, é essencial considerar a pontuação combinada de múltiplas camadas de análise, além de entender as limitações e o comportamento de cada modelo empregado.

O sistema foi projetado para ser modular, permitindo a adição de novos modelos ou heurísticas no futuro, aumentando a confiabilidade da detecção. Não confie apenas em um único fator; a força do Olho de Thoth está na união de diferentes visões.

---

## 📦 Requisitos

Para rodar localmente:

```bash
pip install opencv-python numpy mediapipe==0.10.13 tensorflow pytesseract pillow scikit-learn scipy
```

💡 **No Windows, instale também:**

- **Tesseract OCR:** [Baixe aqui](https://github.com/tesseract-ocr/tesseract)  
- **FFmpeg:** [Baixe aqui](https://ffmpeg.org/) e adicione ao **PATH**
> Os arquivos estão na pasta *recursos necessários* junto com o *tutorial.txt*
---

## 🚀 Como usar

Clone o repositório:

```bash
git clone https://github.com/DaviMattosDev/Olho-de-Thoth.git
cd Olho-de-Thoth
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute:

```bash
python main.py
```

---

## 🌐 Para Rodar no Google Colab:

### 1º célula - bibliotecas

```bash
!pip install numpy
!pip install mediapipe==0.10.9
!pip install opencv-python-headless
!pip install tensorflow
```

### 2º célula - código

Implemente via `main_colab_google.ipynb` ou faça upload e execute o script principal.

---

## 📈 Exemplo de saída

```txt
👁️‍🗨️ RESULTADO DA ANÁLISE
----------------------------------------
Problemas faciais detectados: 0  
Piscadelas detectadas: 0  
Nitidez média: 257.40  
Tremores médios: 3.24  
Média de frequência (FFT): 25.87  
✅ Respiração natural detectada  
Marca(s) d'água detectada(s): Sim  
Probabilidade média de IA: 47.79%  
Detecção por modelo:  
 - Xception: 0.48%  
Palavras-chave suspeitas: ai, google, ai  
Pontuação final: 2.0 / 8  
Status: ✅ Provavelmente real  

--- METADADOS ---  
{ ... }
```

---

## 🧬 Tecnologias Usadas

- 🐍 Python 3.x  
- 📹 OpenCV  
- 🧮 NumPy  
- 👁️ MediaPipe (FaceMesh)  
- 🤖 TensorFlow + Keras  
- 📊 FFT (transformada de Fourier)  
- 📝 JSON + FFmpeg  
- 🖋️ Pytesseract (OCR)  

---

## 🧩 Futuras melhorias

- 📄 Exportar relatório em PDF após análise  
- 📊 Suporte a batch de vídeos  
- 📈 Comparação de resultados entre múltiplos modelos  
- 🖥️ Interface gráfica (Tkinter ou PySide/PyQt)  
- 📁 Histórico de análises salvas localmente  

---

## 📚 Referências

- Chollet, F. (2016). *Xception: Deep Learning with Depthwise Separable Convolutions*  
- Rossler, A. et al. (2019). *FaceForensics++*  
- MediaPipe FaceMesh – GitHub  
- FFmpeg Documentation – [FFmpeg](https://ffmpeg.org/documentation.html)

---

## 🙏 Créditos

Desenvolvido por **[Davi Mattos]**  
Inspiração: **Thoth**, o deus da sabedoria e da verdade.

---

## 📌 Licença

**MIT License** – veja o arquivo LICENSE para mais informações.
