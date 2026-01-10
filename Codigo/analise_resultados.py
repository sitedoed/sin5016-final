"""
ANÁLISE DOS RESULTADOS DO PROJETO - VERSÃO FINAL
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import os

print("=" * 70)
print("ANÁLISE DOS RESULTADOS - IDENTIFICAÇÃO FACIAL COM CNN E cGAN")
print("=" * 70)

# Carregar resultados
with open('../Resultados/cnn_final_results.json', 'r') as f:
    data = json.load(f)

# Extrair dados
baseline_acc = data['Baseline']['accuracy']
baseline_loss = data['Baseline']['loss']
baseline_samples = data['Baseline']['train_samples']

augmented_acc = data['With_Augmentation']['accuracy']
augmented_loss = data['With_Augmentation']['loss']
augmented_samples = data['With_Augmentation']['train_samples']

num_classes = data['Dataset_Info']['num_classes']
test_samples = data['Dataset_Info']['test_samples']
total_samples = data['Dataset_Info']['total_samples']
val_samples = data['Dataset_Info']['validation_samples']

print(f"\n📈 RESUMO ESTATÍSTICO:")
print("-" * 50)
print(f"📊 DATASET:")
print(f"   • Total de imagens: {total_samples:,}")
print(f"   • Classes: {num_classes}")
print(f"   • Teste: {test_samples:,} imagens")
print(f"   • Validação: {val_samples:,} imagens")

print(f"\n🎯 BASELINE (sem augmentation):")
print(f"   • Acurácia: {baseline_acc:.2%}")
print(f"   • Loss: {baseline_loss:.4f}")
print(f"   • Amostras de treino: {baseline_samples:,}")

print(f"\n🚀 COM AUGMENTATION (cGAN):")
print(f"   • Acurácia: {augmented_acc:.2%}")
print(f"   • Loss: {augmented_loss:.4f}")
print(f"   • Amostras de treino: {augmented_samples:,}")
print(f"   • Imagens geradas: {augmented_samples - baseline_samples:,}")

# Calcular diferenças
acc_diff = augmented_acc - baseline_acc
loss_diff = augmented_loss - baseline_loss
acc_improvement = (acc_diff / baseline_acc) * 100
samples_increase = ((augmented_samples - baseline_samples) / baseline_samples) * 100

print(f"\n📊 COMPARAÇÃO:")
print("-" * 50)
print(f"   • Diferença de acurácia: {acc_diff:+.4f} ({acc_improvement:+.2f}%)")
print(f"   • Diferença de loss: {loss_diff:+.4f}")
print(f"   • Aumento no dataset: {augmented_samples - baseline_samples:,} imagens ({samples_increase:+.1f}%)")

print(f"\n🔍 INTERPRETAÇÃO DOS RESULTADOS:")
print("-" * 50)

if acc_improvement > 2:
    print("   ✅ A augmentation com cGAN melhorou SIGNIFICATIVAMENTE o desempenho.")
    print("   As imagens sintéticas foram de alta qualidade e úteis para o modelo.")
elif acc_improvement > 0.5:
    print("   🔄 A augmentation com cGAN teve um efeito POSITIVO, porém MODESTO.")
    print("   As imagens geradas ajudaram, mas o impacto foi limitado.")
elif acc_improvement > -0.5:
    print("   ⚠ A augmentation com cGAN teve impacto NEUTRO/INSIGNIFICANTE.")
    print("   Possíveis causas:")
    print("   1. Poucas imagens geradas (apenas 92 de 5,556)")
    print("   2. Qualidade limitada das imagens sintéticas")
    print("   3. Dataset já suficientemente diversificado")
else:
    print("   ❌ A augmentation com cGAN PIOROU o desempenho.")
    print("   As imagens sintéticas podem ter introduzido ruído ou padrões enganosos.")

print(f"\n💡 RECOMENDAÇÕES PARA MELHORIA:")
print("-" * 50)
print("   1. 📈 Aumentar treinamento da cGAN (200+ épocas)")
print("   2. 🖼️  Aumentar resolução (128x128 ou 256x256)")
print("   3. 🔄 Combinar com augmentation tradicional")
print("   4. ⚖️  Balancear dataset antes de gerar novas imagens")
print("   5. 🎯 Usar dataset CelebA real (202k imagens)")

# ================= GRÁFICOS =================
print(f"\n🎨 GERANDO VISUALIZAÇÕES...")
print("-" * 50)

# 1. GRÁFICO PRINCIPAL - Comparação lado a lado
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('ANÁLISE COMPARATIVA: CNN COM E SEM AUGMENTATION POR cGAN', 
             fontsize=16, fontweight='bold', y=1.02)

# 1.1 Acurácia
models = ['Baseline\n(sem augmentation)', 'Com Augmentation\n(cGAN)']
accuracies = [baseline_acc, augmented_acc]
colors_acc = ['#3498db', '#2ecc71']

bars1 = ax1.bar(models, accuracies, color=colors_acc, alpha=0.85, 
                edgecolor='black', linewidth=1.5, width=0.7)
ax1.set_ylabel('Acurácia', fontsize=12, fontweight='bold')
ax1.set_title('Comparação de Acurácia no Conjunto de Teste', 
              fontsize=14, fontweight='bold', pad=15)
ax1.set_ylim([0.70, 0.78])
ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
ax1.tick_params(axis='x', rotation=15)

# Adicionar valores
for bar, acc in zip(bars1, accuracies):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.002,
            f'{acc:.2%}', ha='center', va='bottom', 
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

# 1.2 Loss
losses = [baseline_loss, augmented_loss]
colors_loss = ['#e74c3c', '#f39c12']

bars2 = ax2.bar(models, losses, color=colors_loss, alpha=0.85,
                edgecolor='black', linewidth=1.5, width=0.7)
ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax2.set_title('Comparação de Loss no Conjunto de Teste',
              fontsize=14, fontweight='bold', pad=15)
ax2.set_ylim([0.70, 0.74])
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
ax2.tick_params(axis='x', rotation=15)

for bar, loss in zip(bars2, losses):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.002,
            f'{loss:.4f}', ha='center', va='bottom',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

# 1.3 Impacto da Augmentation
improvement_data = [acc_improvement]
colors_imp = ['#9b59b6' if acc_improvement >= 0 else '#e74c3c']

bars3 = ax3.bar(['Impacto da cGAN'], improvement_data, color=colors_imp, alpha=0.85,
                edgecolor='black', linewidth=1.5, width=0.6)
ax3.set_ylabel('Variação Percentual (%)', fontsize=12, fontweight='bold')
ax3.set_title('Impacto da Augmentation por cGAN na Acurácia',
              fontsize=14, fontweight='bold', pad=15)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
ax3.set_ylim([-1, 1])

for bar, imp in zip(bars3, improvement_data):
    height = bar.get_height()
    va_pos = 'bottom' if imp >= 0 else 'top'
    y_offset = 0.02 if imp >= 0 else -0.02
    ax3.text(bar.get_x() + bar.get_width()/2., height + y_offset,
            f'{imp:+.2f}%', ha='center', va=va_pos,
            fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

# 1.4 Tamanho do Dataset
train_sizes = [baseline_samples, augmented_samples]
colors_sizes = ['#3498db', '#2ecc71']

bars4 = ax4.bar(models, train_sizes, color=colors_sizes, alpha=0.85,
                edgecolor='black', linewidth=1.5, width=0.7)
ax4.set_ylabel('Número de Imagens', fontsize=12, fontweight='bold')
ax4.set_title('Tamanho do Conjunto de Treino',
              fontsize=14, fontweight='bold', pad=15)
ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
ax4.tick_params(axis='x', rotation=15)

for bar, size in zip(bars4, train_sizes):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 20,
            f'{size:,}', ha='center', va='bottom',
            fontsize=11, fontweight='bold')

# Linha de aumento
increase = augmented_samples - baseline_samples
mid_y = baseline_samples + increase/2
ax4.annotate('', xy=(0.85, mid_y), xytext=(0.15, mid_y),
             arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax4.text(0.5, mid_y + 30, f'+{increase} imagens\n(+{samples_increase:.1f}%)',
         ha='center', va='bottom', fontsize=10, color='red', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

plt.tight_layout()
plt.savefig('../Resultados/analise_comparativa_completa.png', 
            dpi=150, bbox_inches='tight', facecolor='white')
print(f"✅ Gráfico 1 salvo: ../Resultados/analise_comparativa_completa.png")

# 2. GRÁFICO - Distribuição do Dataset
fig2, (ax2_1, ax2_2) = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle('DISTRIBUIÇÃO DO DATASET E IMPACTO DA cGAN', 
              fontsize=16, fontweight='bold', y=1.02)

# 2.1 Distribuição
sizes = [baseline_samples, val_samples, test_samples]
labels = [f'Treino\n{baseline_samples:,}', 
          f'Validação\n{val_samples:,}', 
          f'Teste\n{test_samples:,}']
colors_dist = ['#3498db', '#f39c12', '#e74c3c']
explode = (0.05, 0, 0)

wedges1, texts1, autotexts1 = ax2_1.pie(sizes, labels=labels, colors=colors_dist,
                                        autopct='%1.1f%%', startangle=90,
                                        explode=explode, shadow=True)
ax2_1.set_title('Distribuição do Dataset Completo', fontsize=14, fontweight='bold')

# 2.2 Comparação Treino vs Treino+Aumentado
train_comparison = [baseline_samples, augmented_samples - baseline_samples]
train_labels = [f'Imagens Originais\n{baseline_samples:,}',
                f'Geradas por cGAN\n{augmented_samples - baseline_samples:,}']
train_colors = ['#3498db', '#2ecc71']
train_explode = (0.05, 0)

wedges2, texts2, autotexts2 = ax2_2.pie(train_comparison, labels=train_labels, 
                                        colors=train_colors, autopct='%1.1f%%',
                                        startangle=90, explode=train_explode, shadow=True)
ax2_2.set_title('Composição do Conjunto de Treino (com augmentation)', 
                fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('../Resultados/distribuicao_dataset.png', 
            dpi=150, bbox_inches='tight', facecolor='white')
print(f"✅ Gráfico 2 salvo: ../Resultados/distribuicao_dataset.png")

# 3. GRÁFICO - Tabela Resumo
fig3, ax3_table = plt.subplots(figsize=(12, 5))
ax3_table.axis('tight')
ax3_table.axis('off')

# Dados da tabela
table_data = [
    ["Métrica", "Baseline (sem augmentation)", "Com Augmentation (cGAN)", "Diferença / Impacto"],
    ["Acurácia", f"{baseline_acc:.2%}", f"{augmented_acc:.2%}", 
     f"{acc_diff:+.4f} ({acc_improvement:+.2f}%)"],
    ["Loss", f"{baseline_loss:.4f}", f"{augmented_loss:.4f}", 
     f"{loss_diff:+.4f}"],
    ["Amostras de Treino", f"{baseline_samples:,}", f"{augmented_samples:,}", 
     f"+{augmented_samples - baseline_samples:,} ({samples_increase:+.1f}%)"],
    ["Classes", f"{num_classes}", f"{num_classes}", "Igual"],
    ["Épocas Treinadas", f"{data['Baseline']['epochs_trained']}", 
     f"{data['With_Augmentation']['epochs_trained']}", "Igual"]
]

# Criar tabela
table = ax3_table.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# Estilizar tabela
for i in range(len(table_data)):
    for j in range(len(table_data[0])):
        cell = table[i, j]
        
        # Cabeçalho
        if i == 0:
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', weight='bold', fontsize=12)
        
        # Linhas de dados
        else:
            if j == 3:  # Coluna de diferença
                if acc_improvement >= 0:
                    cell.set_facecolor('#d5f4e6')  # Verde claro para positivo
                else:
                    cell.set_facecolor('#f4d5d5')  # Vermelho claro para negativo
                cell.set_text_props(weight='bold')
            
            # Alternar cores nas linhas
            elif i % 2 == 0:
                cell.set_facecolor('#f8f9fa')
            else:
                cell.set_facecolor('#e9ecef')

ax3_table.set_title('TABELA RESUMO - COMPARAÇÃO DE RESULTADOS', 
                    fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('../Resultados/tabela_resumo_detalhada.png', 
            dpi=150, bbox_inches='tight', facecolor='white')
print(f"✅ Gráfico 3 salvo: ../Resultados/tabela_resumo_detalhada.png")

# 4. GRÁFICO - Métricas de Desempenho (radar chart)
fig4 = plt.figure(figsize=(10, 8))
ax4_radar = plt.subplot(111, polar=True)

# Métricas normalizadas (0-1)
categories = ['Acurácia', '1-Loss', 'Tamanho Dataset', 'Impacto']
N = len(categories)

# Normalizar valores
acc_norm = augmented_acc  # Já está entre 0-1
loss_norm = 1 - (augmented_loss / 5)  # Normalizar loss (supondo max 5)
size_norm = augmented_samples / 10000  # Normalizar pelo máximo esperado
impact_norm = (acc_improvement + 5) / 10  # Mapear [-5%, +5%] para [0, 1]

values = [acc_norm, loss_norm, size_norm, impact_norm]
values += values[:1]  # Fechar o polígono

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# Desenhar gráfico radar
ax4_radar.plot(angles, values, linewidth=2, linestyle='solid', color='#2ecc71')
ax4_radar.fill(angles, values, alpha=0.25, color='#2ecc71')
ax4_radar.set_xticks(angles[:-1])
ax4_radar.set_xticklabels(categories, fontsize=12, fontweight='bold')
ax4_radar.set_ylim(0, 1)
ax4_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax4_radar.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=10)
ax4_radar.grid(True, alpha=0.3)

# Adicionar valores
for angle, value, label in zip(angles[:-1], values[:-1], categories):
    ax4_radar.text(angle, value + 0.05, f'{value:.2f}', 
                   ha='center', va='center', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax4_radar.set_title('GRÁFICO RADAR - DESEMPENHO DO MODELO COM AUGMENTATION', 
                    fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('../Resultados/grafico_radar_desempenho.png', 
            dpi=150, bbox_inches='tight', facecolor='white')
print(f"✅ Gráfico 4 salvo: ../Resultados/grafico_radar_desempenho.png")

print("\n" + "=" * 70)
print("📋 RESUMO FINAL PARA RELATÓRIO")
print("=" * 70)

print(f"\n✅ PONTOS FORTES DO PROJETO:")
print("1. ✅ Pipeline completo implementado (cGAN → CNN → Avaliação)")
print("2. ✅ cGAN funcional - gerou 92 imagens sintéticas válidas")
print("3. ✅ CNN eficiente - 75%+ de acurácia com dados simulados")
print("4. ✅ Metodologia sólida - divisão adequada dos dados")
print("5. ✅ Análise comparativa - baseline vs augmentation")

print(f"\n⚠ LIMITAÇÕES IDENTIFICADAS:")
print("1. ⚠ Dataset simulado (não CelebA real)")
print("2. ⚠ Impacto limitado da cGAN (+0.33% apenas)")
print("3. ⚠ Poucas imagens geradas (1.66% de aumento)")
print("4. ⚠ Baixa resolução (64x64 pixels)")

print(f"\n🎯 CONTRIBUIÇÕES PARA O RELATÓRIO:")
print("1. 📊 Demonstração prática de cGAN para augmentation")
print("2. 📈 Análise quantitativa do impacto da augmentation")
print("3. 🔍 Identificação de limitações e gargalos")
print("4. 💡 Proposta de melhorias para trabalhos futuros")

print(f"\n📊 MÉTRICAS CHAVE (para destacar no relatório):")
print(f"• 🎯 Acurácia Baseline: {baseline_acc:.2%}")
print(f"• 🚀 Acurácia com cGAN: {augmented_acc:.2%}")
print(f"• 📈 Melhoria: {acc_improvement:+.2f}%")
print(f"• 🖼️  Imagens geradas: {augmented_samples - baseline_samples:,}")
print(f"• 👥 Classes identificadas: {num_classes}")
print(f"• 📦 Tamanho total do dataset: {total_samples:,} imagens")

print("\n" + "=" * 70)
print("✅ ANÁLISE CONCLUÍDA COM SUCESSO!")
print("=" * 70)
print("\n📁 ARQUIVOS GERADOS PARA SEU RELATÓRIO:")
print("  • ../Resultados/analise_comparativa_completa.png")
print("  • ../Resultados/distribuicao_dataset.png")
print("  • ../Resultados/tabela_resumo_detalhada.png")
print("  • ../Resultados/grafico_radar_desempenho.png")
print("\n📄 ARQUIVOS EXISTENTES:")
print("  • ../Resultados/cnn_final_results.json")
print("  • ../Resultados/cnn_baseline_final.h5")
print("  • ../Resultados/cnn_augmented_final.h5")
print("  • ../Resultados/cgan_generator.h5")
print("  • ../Resultados/cnn_history_*.png")
print("=" * 70)

# Tentar mostrar gráficos
try:
    plt.show()
except:
    print("\n⚠ Gráficos salvos mas não mostrados (ambiente não interativo)")
    print("📌 Para visualizar: abra os arquivos .png na pasta Resultados/")