# Sistema de Identificação Facial com CNN – Dataset CelebA

Este repositório contém a implementação, experimentos e análise de um **Sistema de Identificação Biométrica Facial** baseado em **Redes Neurais Convolucionais (CNNs)**, utilizando o dataset **CelebA (Celebrities Attributes Dataset)**.

O projeto foi desenvolvido com fins **acadêmicos**, avaliando o desempenho de CNNs em cenários controlados e em larga escala, com foco em **escalabilidade**, **acurácia Top-1 e Top-5**, e **custo computacional**.

---

## 📌 Objetivos do Projeto

- Implementar uma CNN para **identificação facial multi-classe**
- Avaliar desempenho em:
  - Cenário controlado (72 classes)
  - Cenário em larga escala (1.687 classes)
- Comparar impacto do número de épocas no desempenho
- Analisar limitações e propor melhorias arquiteturais
- Produzir documentação técnica clara e reprodutível

---

## 📂 Dataset

**CelebA – Celebrities Attributes Dataset**

- Total de imagens utilizadas: **50.648**
- Total de identidades: **1.687**
- Resolução original do CelebA: **178 × 218 pixels**
- Resolução utilizada no projeto: **64 × 64 pixels**
- Formato: **Grayscale (1 canal)**

### Pré-processamento
- Redimensionamento para 64×64
- Conversão para escala de cinza
- Normalização dos pixels
- Divisão estratificada em treino, validação e teste

---

## 🧪 Divisão dos Experimentos

### 🔹 Experimento Controlado
- Classes: **72**
- Total de imagens: **7.938**
- Treino: 70%
- Validação: 15%
- Teste: 15%

### 🔹 Experimento em Larga Escala
- Classes: **1.687**
- Treino: 35.451 imagens
- Validação: 7.599 imagens
- Teste: 7.598 imagens

---

## 🧠 Arquitetura da CNN

- Entrada: `(64, 64, 1)`
- 4 blocos:
  - `Conv2D`
  - `Batch Normalization`
  - `MaxPooling`
  - `Dropout`
- Camada densa final: **512 neurônios**
- Saída: `Softmax (N_classes)`

### Configuração de Treinamento
- Função de perda: **Categorical Cross-Entropy**
- Otimizador: **Adam**
- Learning rate: **0.001**
- Batch size: **32**
- Épocas testadas: **10 e 30**

---

## 📊 Resultados Principais

### Tabela Comparativa dos Cenários

| Cenário                  | Classes | Épocas | Top-1 (%) | Top-5 (%) | Tempo de Treino |
|--------------------------|---------|--------|-----------|-----------|-----------------|
| Controlado (Baseline)    | 72      | 30     | 75,23     | —         | ~15 min         |
| Controlado + Augmentation| 72      | 30     | 0,08*     | —         | ~16 min         |
| Larga Escala             | 1.687   | 10     | 39,73     | 59,23     | 36 min          |
| Larga Escala             | 1.687   | 30     | 52,51     | 69,35     | 108 min         |

\* Resultado anômalo – provável erro na implementação do data augmentation.

---

## 📈 Análise dos Resultados

- Aumento de **12,78% na Top-1 accuracy** ao passar de 10 para 30 épocas
- **Top-5 accuracy de 69,35%** demonstra aprendizado discriminativo robusto
- Redução esperada de desempenho ao escalar de 72 para 1.687 classes
- Relação **tempo × desempenho** favorável para 30 épocas

---

## ⚠️ Limitações Identificadas

### Técnicas
- Erro crítico no pipeline de data augmentation
- Arquitetura CNN simples para identificação em larga escala
- Softmax não ideal para grande número de classes

### Computacionais
- Treinamento demanda GPU
- Tempo cresce linearmente com o número de épocas

---

## 🚀 Melhorias Propostas

### Curto Prazo
- Correção do pipeline de data augmentation
- Validação rigorosa dos labels após transformação
- Uso de learning rate scheduling

### Médio e Longo Prazo
- Substituição da CNN por:
  - ResNet
  - EfficientNet
- Aprendizado Métrico:
  - Triplet Loss
  - ArcFace
- Uso de embeddings faciais e classificação por similaridade
- Ensemble de modelos

---

## 🏁 Conclusões

- O sistema alcançou **52,51% de acurácia Top-1** para **1.687 identidades**
- Top-5 accuracy de **69,35%** indica potencial prático
- Arquitetura é funcional, mas não ideal para produção
- Projeto fornece base sólida para evolução futura

---

## 🛠️ Tecnologias Utilizadas

- Python 3.x
- TensorFlow / Keras
- NumPy
- OpenCV
- Matplotlib
- Dataset CelebA

---

## 👤 Autor

**Edson de Oliveira Vieira**  
Programa de Pós-Graduação – Universidade de São Paulo (USP)

**Orientador:**  
Prof. Dr. Clodoaldo A. Lima

---

## 📅 Data

Projeto desenvolvido e avaliado em **Janeiro de 2026**.

---

## 📎 Observação Final

Este repositório possui caráter **acadêmico e experimental**.  
Os resultados **não devem ser utilizados diretamente em sistemas críticos de produção sem validações adicionais**.
