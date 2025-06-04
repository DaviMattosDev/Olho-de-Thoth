def get_model_description(model_name):
    # --------------------------
    #Retorna uma string com a descrição detalhada de cada modelo de IA.
    # --------------------------
    descriptions = {
        "Xception": (
            "📘 XCEPTION (Extreme Inception)\n"
            "================================\n"
            "• Arquitetura: Evolução do Inception V3 com convoluções separáveis em profundidade\n"
            "• Parâmetros: ~23 milhões\n"
            "• Profundidade: 126 camadas\n"
            "• Precisão ImageNet: 94.5% (Top-5)\n\n"
            "VANTAGENS PARA DEEPFAKE:\n"
            "✓ Excelente em detectar artefatos de compressão e inconsistências de textura\n"
            "✓ Sensível a padrões de alta frequência alterados em manipulações faciais\n"
            "✓ Ótimo para vídeos HD/4K com detalhes sutis\n"
            "✓ Detecta bem bordas artificiais e transições não naturais\n\n"
            "QUANDO USAR:\n"
            "• Vídeos de alta qualidade (1080p+)\n"
            "• Suspeita de deepfakes sofisticados\n"
            "• Quando precisão é mais importante que velocidade"
        ),
        
        "EfficientNet": (
            "⚡ EFFICIENTNET-B0\n"
            "==================\n"
            "• Arquitetura: Rede neural com escalonamento composto otimizado\n"
            "• Parâmetros: ~5.3 milhões (B0)\n"
            "• Profundidade: Variável (B0-B7)\n"
            "• Precisão ImageNet: 93.3% (Top-5)\n\n"
            "VANTAGENS PARA DEEPFAKE:\n"
            "✓ Melhor relação precisão/velocidade do mercado\n"
            "✓ Detecta padrões em múltiplas escalas simultaneamente\n"
            "✓ Eficiente em recursos computacionais (30% mais rápido)\n"
            "✓ Boa generalização para diferentes tipos de manipulação\n\n"
            "QUANDO USAR:\n"
            "• Análise em tempo real ou quase real\n"
            "• Dispositivos com recursos limitados\n"
            "• Grande volume de vídeos para processar\n"
            "• Primeira triagem rápida"
        ),
        
        "Inception": (
            "🌀 INCEPTION V3\n"
            "===============\n"
            "• Arquitetura: Módulos Inception com filtros paralelos de múltiplos tamanhos\n"
            "• Parâmetros: ~24 milhões\n"
            "• Profundidade: 159 camadas\n"
            "• Precisão ImageNet: 93.7% (Top-5)\n\n"
            "VANTAGENS PARA DEEPFAKE:\n"
            "✓ Captura características em múltiplas escalas (1x1, 3x3, 5x5)\n"
            "✓ Excelente para detectar inconsistências temporais\n"
            "✓ Forte em identificar artefatos de blending facial\n"
            "✓ Detecta bem distorções geométricas sutis\n\n"
            "QUANDO USAR:\n"
            "• Vídeos com movimentos complexos\n"
            "• Deepfakes com múltiplas faces\n"
            "• Quando há suspeita de manipulação temporal\n"
            "• Análise detalhada frame a frame"
        ),
        
        "ResNet": (
            "🛡️ RESNET-50\n"
            "=============\n"
            "• Arquitetura: Rede residual com conexões de atalho (skip connections)\n"
            "• Parâmetros: ~25.6 milhões\n"
            "• Profundidade: 50 camadas (versões: 18, 34, 50, 101, 152)\n"
            "• Precisão ImageNet: 92.1% (Top-5)\n\n"
            "VANTAGENS PARA DEEPFAKE:\n"
            "✓ Extremamente robusto contra degradação de gradiente\n"
            "✓ Captura características profundas e abstratas\n"
            "✓ Resistente a ruído e compressão de vídeo\n"
            "✓ Excelente em detectar inconsistências de iluminação\n"
            "✓ Forte contra adversarial attacks\n\n"
            "QUANDO USAR:\n"
            "• Vídeos de baixa qualidade ou comprimidos\n"
            "• Deepfakes que tentam enganar outros detectores\n"
            "• Análise de consistência de iluminação/sombras\n"
            "• Quando robustez é prioridade"
        )
    }
    
    # Adicionar informação comparativa geral
    general_info = (
        "\n\n💡 DICA DE ESCOLHA:\n"
        "-------------------\n"
        "• Velocidade: EfficientNet > ResNet > Inception > Xception\n"
        "• Precisão: Xception ≈ Inception > ResNet > EfficientNet\n"
        "• Robustez: ResNet > Xception > Inception > EfficientNet\n"
        "• Uso de memória: EfficientNet < ResNet < Inception < Xception"
    )
    
    description = descriptions.get(model_name, f"ℹ️ Nenhuma descrição disponível para '{model_name}'.")
    
    # Adicionar informação geral apenas se encontrou a descrição
    if model_name in descriptions:
        description += general_info
    
    return description