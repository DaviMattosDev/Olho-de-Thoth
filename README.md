---
# Olho de Thoth
<div align="center">
  <img src="https://media1.tenor.com/m/DvFsvcuRZC0AAAAd/ra-amun.gif" alt="Olho-de-thoth" height='300px' width='400px'>
</div>

🧠 "O Olho de Thoth vê além da ilusão."

Um sistema híbrido para detectar vídeos provavelmente gerados por inteligência artificial — usando análise visual, aprendizado de máquina e metadados.

## 🌟 Sobre

Olho-de-Thoth é uma ferramenta inspirada na sabedoria do deus egípcio Thoth, capaz de "ver" se um vídeo foi criado por IA ou gravado na vida real. Ele usa técnicas avançadas de visão computacional, modelos pré-treinados (como XceptionNet), análise de artefatos visuais e leitura de metadados para identificar pistas sutis de vídeos sintéticos.

Com a explosão de modelos como Google Veo 3, OpenAI Sora e Stable Video Diffusion, torna-se essencial ter sistemas capazes de distinguir o real do falso. O Olho-de-Thoth nasce dessa necessidade: ser o olho que enxerga a verdade por trás da tecnologia.

## 🔍 Funcionalidades

- ✅ Análise facial: detecta inconsistências faciais e ausência de piscadelas
- 🤖 Uso de modelo de IA (XceptionNet) para classificação frame a frame
- 📸 Detecção de nitidez anormal
- 🎞️ Análise de tremor entre frames (jitter)
- 📊 Análise de frequência espacial (FFT)
- 📄 Leitura de metadados do vídeo (via FFmpeg / ffprobe)
- 🧮 Sistema de pontuação final para decisão

## 🧪 Como funciona?

O detector analisa o vídeo em múltiplas camadas:

1. Extração de frames do vídeo
2. Análise facial e microcomportamentos
3. Medição de nitidez e jitter
4. Transformada de Fourier (FFT) para padrões espaciais
5. Classificação com modelo XceptionNet
6. Leitura de metadados do arquivo
7. Atribuição de pontuação e decisão final

🔍 Se a soma das pistas for maior ou igual a 4 → Provavelmente gerado por IA

## 📦 Requisitos

Para rodar localmente:

```bash
pip install opencv-python numpy mediapipe tensorflow ffmpeg-python
```

💡 No Windows, instale também o FFmpeg e adicione ao PATH.

## 🚀 Como usar

Clone o repositório:

```bash
git clone https://github.com/seu-usuario/Olho-de-Thoth.git 
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

Digite o caminho completo do vídeo quando solicitado:

```bash
Digite o caminho completo do vídeo (ex: ./video.mp4): videos/exemplo.mp4
```

Para Rodar no Google Colab:

1º célula - bibliotecas
```bash
!pip install numpy==1.23.5
!pip install --upgrade --no-cache-dir mediapipe
!pip install opencv-python-headless
```

2º célula - código
```bash
copie ou importe o main_colab_google.ipynb
```


## 📈 Exemplo de saída

```
--- RESULTADO DA ANÁLISE ---
Problemas faciais detectados: 3
Piscadelas detectadas: 0
Nitidez média: 26.32
Média de jitter: 4.92
Média de frequência (FFT): 147.07
Probabilidade média de IA: 57.00%
[Pista encontrada nos metadados]: google, lavf

Pontuação total de suspeita (limiar: 4): 6
⚠️ Resultado: Provavelmente gerado por IA
```

## 🧬 Tecnologias Usadas

- 🐍 Python 3.x
- 📹 OpenCV
- 🧮 NumPy
- 👁️ MediaPipe (FaceMesh)
- 🤖 TensorFlow + Keras
- 📊 FFT (transformada de Fourier)
- 📝 JSON + FFmpeg

## 🧩 Futuras melhorias

- 📷 Adicionar OCR para detectar marcas d'água invisíveis
- 🫁 Detectar respiração natural
- 📁 Exportar relatório em PDF após análise
- 🖥️ Interface gráfica (Tkinter ou PySide)
- 📊 Suporte a batch de vídeos

## 📚 Referências

- Chollet, F. (2016). Xception: Deep Learning with Depthwise Separable Convolutions
- Rossler, A. et al. (2019). FaceForensics++
- MediaPipe FaceMesh – [GitHub](https://github.com/google/mediapipe)
- FFmpeg Documentation – [FFmpeg](https://ffmpeg.org/)

## 🙏 Créditos

Desenvolvido por [Davi Mattos].  
Inspiração: Thoth, o deus da sabedoria e da verdade.

## 📌 Licença

MIT License – veja LICENSE para mais informações.

---
