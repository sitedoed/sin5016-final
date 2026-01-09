Projeto de Identificação Biométrica Facial com CNN
📋 Sobre o Projeto

Este projeto implementa um sistema de identificação biométrica facial utilizando Redes Neurais Convolucionais (CNNs) para aprendizado de representações discriminativas diretamente dos dados brutos. O sistema é treinado e avaliado utilizando um subconjunto da base de dados CelebA.

Objetivo Geral: Desenvolver um sistema robusto de identificação facial baseado em CNNs capaz de reconhecer identidades em condições variadas de iluminação, expressão e pose.
🎯 Objetivos Específicos

    Implementar uma CNN para reconhecimento facial

    Avaliar o impacto de diferentes técnicas de pré-processamento

    Comparar resultados com abordagens tradicionais

    Analisar desempenho, acurácia e limitações do sistema

    Implementar aumento de dados com cGAN (opcional)

📊 Dataset
CelebA Subset

    Origem: CelebA (Celebrities Attributes Dataset)

    Tamanho original: 202.599 imagens (10.177 identidades)

    Subconjunto utilizado: 20% da base original (≈40.520 imagens)

    Resolução: 64×64 pixels (otimizado de trabalho anterior)

    Formato: Grayscale (1 canal)

    Distribuição: ≈2.000 identidades, média de 20 imagens por identidade

Divisão dos Dados

    Treino: 70% (≈28.364 imagens)

    Validação: 15% (≈6.078 imagens)

    Teste: 15% (≈6.078 imagens)

🏗️ Arquitetura do Sistema
1. Pré-processamento

    Redimensionamento para 64×64 pixels

    Normalização de pixels para [0, 1]

    Data augmentation (flip horizontal, rotações leves, ajuste de brilho)

    One-hot encoding dos rótulos

2. Arquitetura CNN Principal
text

Camada de Entrada: (64, 64, 1)
├── Conv2D(32, 3×3) + ReLU + BatchNorm
├── MaxPooling2D(2×2) + Dropout(0.25)
├── Conv2D(64, 3×3) + ReLU + BatchNorm
├── MaxPooling2D(2×2) + Dropout(0.25)
├── Conv2D(128, 3×3) + ReLU + BatchNorm
├── MaxPooling2D(2×2) + Dropout(0.25)
├── Conv2D(256, 3×3) + ReLU + BatchNorm
├── MaxPooling2D(2×2) + Dropout(0.25)
├── Flatten()
├── Dense(512) + ReLU + BatchNorm + Dropout(0.5)
└── Dense(N_classes) + Softmax

3. Configuração de Treinamento

    Função de perda: Categorical Cross-Entropy

    Otimizador: Adam (learning_rate=0.001)

    Métricas: Acurácia, Precision, Recall, F1-Score

    Batch size: 32 ou 64

    Épocas: Até early stopping (paciência=10)

4. cGAN para Data Augmentation (Opcional)

    Geração de imagens sintéticas condicionadas por identidade

    Balanceamento de classes minoritárias

    Arquitetura DCGAN modificada para grayscale

📁 Estrutura do Projeto
text

projeto_facial/
├── data/                          # Dados e datasets
├── notebooks/                     # Análises exploratórias
├── src/                           # Código fonte
│   ├── data/                      # Manipulação de dados
│   ├── models/                    # Definição dos modelos
│   ├── training/                  # Treinamento
│   ├── evaluation/                # Avaliação
│   └── utils/                     # Utilitários
├── configs/                       # Configurações
├── experiments/                   # Resultados experimentais
├── reports/                       # Relatórios
├── scripts/                       # Scripts executáveis
└── outputs/                       # Saídas finais

🚀 Como Executar
Pré-requisitos
bash

Python 3.8+
TensorFlow 2.8+
OpenCV
scikit-learn
matplotlib
numpy
pandas

Instalação
bash

# Clonar repositório
git clone https://github.com/seu-usuario/projeto-facial-cnn.git
cd projeto-facial-cnn

# Criar ambiente virtual
python -m venv final
source final/bin/activate  # Linux/Mac
# ou
final\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt

Execução do Pipeline
bash

# 1. Pré-processamento dos dados
python scripts/run_preprocessing.py --input_dir data/raw --output_dir data/processed

# 2. Treinamento da CNN
python scripts/train_cnn.py --config configs/cnn_config.yaml

# 3. Avaliação do modelo
python scripts/evaluate_model.py --model_path experiments/model_final.h5 --test_dir data/processed/test

# 4. Geração de relatório (opcional)
python scripts/generate_report.py --output_dir reports/

📈 Métricas de Avaliação

    Acurácia Geral: Top-1 e Top-5 accuracy

    Métricas por Classe: Precision, Recall, F1-Score

    Matriz de Confusão: Análise de erros entre classes

    Curva ROC: Para avaliação multiclasse

    Tempo de Inferência: Performance em tempo real

📊 Resultados Esperados
Métrica	CNN Baseline	CNN + Augmentation	CNN + cGAN
Acurácia (Top-1)	~85%	~88%	~90%
Acurácia (Top-5)	~95%	~97%	~98%
F1-Score Médio	~0.84	~0.87	~0.89
Tempo Inferência	<50ms	<50ms	<50ms
🧪 Experimentos Realizados

    Experimento 1: CNN baseline com pré-processamento mínimo

    Experimento 2: CNN com data augmentation tradicional

    Experimento 3: CNN com aumento de dados via cGAN

    Experimento 4: Transfer learning com EfficientNet

    Experimento 5: Ensemble de modelos

📝 Relatório Técnico

O relatório técnico inclui:

    Revisão bibliográfica sobre reconhecimento facial

    Metodologia detalhada

    Análise comparativa dos experimentos

    Discussão de resultados e limitações

    Propostas de trabalho futuro


🔧 Tecnologias Utilizadas

    Linguagem: Python 3.8+

    Deep Learning: TensorFlow 2.x / Keras

    Processamento de Imagens: OpenCV, PIL

    Análise de Dados: NumPy, Pandas, Matplotlib

    Avaliação: scikit-learn

    Desenvolvimento: Jupyter Notebook, Git

⚠️ Limitações e Desafios

    Variabilidade intra-classe: Expressões, iluminação e poses diferentes

    Similaridade inter-classe: Algumas identidades são visualmente similares

    Balanceamento de classes: Distribuição desigual no dataset original

    Recursos computacionais: Treinamento demanda GPU com memória suficiente

📈 Trabalho Futuro

    Implementar attention mechanisms na CNN

    Explorar arquiteturas mais recentes (Vision Transformers)

    Adicionar reconhecimento de atributos (idade, gênero, emoção)

    Implementar sistema em tempo real com OpenCV

    Testar com outras bases de dados (LFW, VGGFace2)

👥 Autores

    Edson Vieira - Desenvolvimento e análise

    Prof. dr. Clodoaldo A. Lima - Orientação

    Universidade de São Paulo - USP - Suporte institucional

📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo LICENSE para detalhes.
🙏 Agradecimentos

    Universidade de Hong Kong pelo dataset CelebA

    Comunidade TensorFlow/Keras pela documentação

    Google Colab pelos recursos computacionais


Projeto desenvolvido para a disciplina de Aprendizado de Máquina - USP, 2025
