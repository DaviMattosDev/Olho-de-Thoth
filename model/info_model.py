def get_model_description(model_name):
    # --------------------------
    #Retorna uma string com a descrição detalhada de cada modelo de IA.
    # --------------------------
    descriptions = {
        "Xception": (
            "[INFO] 🔍 Modelo: Xception\n"
            "- Baseado em convoluções depthwise separáveis, o que reduz o custo computacional sem sacrificar a precisão.\n"
            "- Especialmente eficaz na detecção de padrões sutis e texturas delicadas, comuns em vídeos manipulados por IA.\n"
            "- Excelente desempenho em vídeos de alta qualidade, como deepfakes realistas, graças à sua capacidade de capturar nuances visuais."
        ),
        "EfficientNet": (
            "[INFO] 🔍 Modelo: EfficientNet\n"
            "- Conjunto de modelos otimizados que escalam largura, profundidade e resolução de forma equilibrada.\n"
            "- Oferece alta precisão com menos parâmetros e menor uso de memória e processamento.\n"
            "- Ideal para aplicações em tempo real ou com recursos limitados, sem perda significativa de desempenho na análise de vídeos."
        ),
        "Inception": (
            "[INFO] 🔍 Modelo: InceptionV3\n"
            "- Usa módulos com múltiplos filtros paralelos para extrair informações em diferentes escalas ao mesmo tempo.\n"
            "- Muito eficaz na análise de vídeos com estruturas complexas e muitos detalhes visuais.\n"
            "- Útil para detectar manipulações sutis em diferentes regiões da imagem simultaneamente."
        ),
        "ResNet": (
            "[INFO] 🔍 Modelo: ResNet50\n"
            "- Arquitetura baseada em blocos residuais, que usam conexões de atalho ('skip connections') para evitar o problema do desaparecimento do gradiente.\n"
            "- Permite redes mais profundas com estabilidade no treinamento.\n"
            "- Indicado para vídeos com alterações discretas, como manipulações leves ou bem disfarçadas, mantendo boa generalização."
        )
    }
    return descriptions.get(model_name, f"[INFO] ❓ Modelo desconhecido: {model_name}")