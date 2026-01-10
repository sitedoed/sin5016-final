#!/usr/bin/env python3
"""
GERAR MATRIZ DE CONFUSÃO - VERSÃO CORRIGIDA
"""

import os
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter

def main():
    print("="*70)
    print("GERANDO MATRIZ DE CONFUSÃO - MODELO TREINADO")
    print("="*70)
    
    # 1. CARREGAR MODELO TREINADO
    print("\n[1/4] Carregando modelo treinado...")
    model_path = '../Resultados/cnn_30epocas_best.h5'
    
    if not os.path.exists(model_path):
        print(f"❌ Modelo não encontrado: {model_path}")
        print("Execute primeiro: python cnn_model.py")
        return
    
    model = keras.models.load_model(model_path)
    print(f"✓ Modelo carregado: {model_path}")
    print(f"  Parâmetros: {model.count_params():,}")
    
    # 2. CARREGAR DADOS DE TESTE
    print("\n[2/4] Carregando dados de teste...")
    data_path = '../Images/Processed/'
    
    X_test = np.load(os.path.join(data_path, 'X_test.npy'))
    y_test = np.load(os.path.join(data_path, 'y_test.npy'))
    
    print(f"✓ Dados carregados:")
    print(f"  Teste: {X_test.shape[0]:,} imagens")
    print(f"  Classes: {len(np.unique(y_test))}")
    print(f"  Formato imagens: {X_test.shape[1:]}")
    
    # 3. FAZER PREVISÕES
    print("\n[3/4] Fazendo previsões no conjunto de teste...")
    
    # Fazer previsões em batches
    batch_size = 64
    y_pred_proba = []
    
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i:i+batch_size]
        batch_pred = model.predict(batch, verbose=0)
        y_pred_proba.append(batch_pred)
        
        if (i // batch_size) % 20 == 0:
            print(f"  Processadas {min(i+batch_size, len(X_test)):,}/{len(X_test):,} imagens")
    
    y_pred_proba = np.concatenate(y_pred_proba, axis=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    print(f"✓ Previsões concluídas!")
    
    # 4. CALCULAR ACURÁCIA
    accuracy = np.mean(y_pred == y_test)
    print(f"\n  Acurácia verificada: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 5. GERAR MATRIZ DE CONFUSÃO PARA ALGUMAS CLASSES
    print("\n[4/4] Gerando análise detalhada...")
    
    # Encontrar classes que realmente aparecem nas previsões
    unique_pred_classes = np.unique(y_pred)
    unique_true_classes = np.unique(y_test)
    
    print(f"  Classes únicas nas previsões: {len(unique_pred_classes)}")
    print(f"  Classes únicas nos labels verdadeiros: {len(unique_true_classes)}")
    
    # 5A. GERAR MATRIZ DE CONFUSÃO COMPLETA (mas vamos salvar em arquivo, não plotar toda)
    print("\n  Calculando matriz de confusão completa...")
    cm_full = confusion_matrix(y_test, y_pred)
    
    # Salvar matriz completa em arquivo (para análise posterior)
    np.save('../Resultados/matriz_confusao_completa.npy', cm_full)
    print(f"  ✓ Matriz completa salva (1687x1687)")
    
    # 5B. GERAR VERSÃO REDUZIDA PARA VISUALIZAÇÃO (top 20 classes por acurácia)
    print("\n  Analisando classes com melhor desempenho...")
    
    # Calcular acurácia por classe
    class_accuracies = {}
    for class_id in np.unique(y_test):
        mask = y_test == class_id
        if np.sum(mask) > 0:  # Evitar divisão por zero
            class_acc = np.mean(y_pred[mask] == y_test[mask])
            class_accuracies[class_id] = {
                'accuracy': class_acc,
                'samples': np.sum(mask),
                'correct': np.sum(y_pred[mask] == y_test[mask])
            }
    
    # Ordenar por acurácia
    sorted_classes = sorted(class_accuracies.items(), 
                           key=lambda x: x[1]['accuracy'], 
                           reverse=True)
    
    # Pegar top 15 classes por acurácia
    top_n = 15
    top_classes = [cls for cls, _ in sorted_classes[:top_n]]
    
    print(f"\n  Top {top_n} classes por acurácia:")
    for i, (class_id, metrics) in enumerate(sorted_classes[:top_n]):
        print(f"    Classe {class_id}: {metrics['accuracy']:.2%} "
              f"({metrics['correct']}/{metrics['samples']} amostras)")
    
    # Filtrar para essas classes
    mask = np.isin(y_test, top_classes)
    y_test_filtered = y_test[mask]
    y_pred_filtered = y_pred[mask]
    
    print(f"\n  Amostras nas top {top_n} classes: {len(y_test_filtered):,}")
    
    # Gerar matriz de confusão para essas classes
    cm_top = confusion_matrix(y_test_filtered, y_pred_filtered, labels=top_classes)
    
    # 6. PLOTAR MATRIZ DE CONFUSÃO (top classes)
    plt.figure(figsize=(12, 10))
    
    # Normalizar por linha (recall)
    cm_normalized = cm_top.astype('float') / cm_top.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Lidar com divisão por zero
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=[f'Classe {c}' for c in top_classes],
                yticklabels=[f'Classe {c}' for c in top_classes],
                cbar_kws={'label': 'Recall (Acurácia por Classe)'})
    
    plt.title(f'Matriz de Confusão - Top {top_n} Classes por Acurácia\n(Recall por Classe)', 
             fontsize=14, pad=20)
    plt.xlabel('Classe Predita', fontsize=12)
    plt.ylabel('Classe Verdadeira', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    
    plt.tight_layout()
    
    # Salvar
    output_path = '../Resultados/matriz_confusao_top15_acuracia.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Matriz de confusão (top por acurácia) salva em: {output_path}")
    
    # 7. GERAR GRÁFICO DE ACURÁCIA POR CLASSE
    plt.figure(figsize=(14, 6))
    
    # Pegar top 30 classes para o gráfico
    top_30 = [cls for cls, _ in sorted_classes[:30]]
    accuracies_30 = [class_accuracies[cls]['accuracy'] for cls in top_30]
    samples_30 = [class_accuracies[cls]['samples'] for cls in top_30]
    
    # Gráfico de barras
    x_pos = np.arange(len(top_30))
    
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # Barras de acurácia
    bars = ax1.bar(x_pos, accuracies_30, alpha=0.7, color='steelblue')
    ax1.set_xlabel('Classe', fontsize=12)
    ax1.set_ylabel('Acurácia', fontsize=12, color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_ylim([0, 1.1])
    ax1.set_title(f'Acurácia por Classe - Top 30', fontsize=14, pad=20)
    
    # Adicionar valores nas barras
    for i, (bar, acc) in enumerate(zip(bars, accuracies_30)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.0%}', ha='center', va='bottom', fontsize=8)
    
    # Segundo eixo para número de amostras
    ax2 = ax1.twinx()
    ax2.plot(x_pos, samples_30, 'ro-', linewidth=2, markersize=6)
    ax2.set_ylabel('Número de Amostras', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Configurar eixos X
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'{c}' for c in top_30], rotation=45, ha='right', fontsize=9)
    
    plt.tight_layout()
    
    accuracy_plot_path = '../Resultados/acuracia_por_classe.png'
    plt.savefig(accuracy_plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico de acurácia por classe salvo em: {accuracy_plot_path}")
    
    # 8. GERAR RELATÓRIO ESTATÍSTICO
    print("\n" + "="*70)
    print("RELATÓRIO ESTATÍSTICO")
    print("="*70)
    
    # Estatísticas gerais
    total_samples = len(y_test)
    correct_predictions = np.sum(y_pred == y_test)
    
    print(f"\nEstatísticas gerais:")
    print(f"  Total de amostras: {total_samples:,}")
    print(f"  Previsões corretas: {correct_predictions:,}")
    print(f"  Acurácia geral: {accuracy:.2%}")
    
    # Distribuição de acurácia por classe
    acc_values = [metrics['accuracy'] for _, metrics in sorted_classes]
    
    print(f"\nDistribuição de acurácia por classe:")
    print(f"  Média: {np.mean(acc_values):.2%}")
    print(f"  Mediana: {np.median(acc_values):.2%}")
    print(f"  Máxima: {np.max(acc_values):.2%}")
    print(f"  Mínima: {np.min(acc_values):.2%}")
    print(f"  Desvio padrão: {np.std(acc_values):.2%}")
    
    # Contar classes por faixa de acurácia
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    
    acc_bins = np.digitize(acc_values, bins) - 1
    
    print(f"\nClasses por faixa de acurácia:")
    for i, label in enumerate(bin_labels):
        count = np.sum(acc_bins == i)
        percentage = count / len(acc_values) * 100
        print(f"  {label}: {count} classes ({percentage:.1f}%)")
    
    # 9. SALVAR RESULTADOS DETALHADOS
    results = {
        'modelo': 'cnn_30epocas_best.h5',
        'dataset': {
            'total_imagens': int(total_samples),
            'classes_unicas': int(len(np.unique(y_test))),
            'acurácia_geral': float(accuracy)
        },
        'estatisticas_acurácia': {
            'média': float(np.mean(acc_values)),
            'mediana': float(np.median(acc_values)),
            'máxima': float(np.max(acc_values)),
            'mínima': float(np.min(acc_values)),
            'desvio_padrão': float(np.std(acc_values))
        },
        'distribuição_acurácia': {
            label: int(count) for label, count in zip(bin_labels, 
                [np.sum(acc_bins == i) for i in range(len(bin_labels))])
        },
        'top_classes_acuracia': [
            {
                'classe': int(cls),
                'acuracia': float(metrics['accuracy']),
                'amostras': int(metrics['samples']),
                'corretas': int(metrics['correct'])
            }
            for cls, metrics in sorted_classes[:20]
        ]
    }
    
    results_path = '../Resultados/analise_detalhada_modelo.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Análise detalhada salva em: {results_path}")
    
    # 10. RESUMO FINAL
    print("\n" + "="*70)
    print("ANÁLISE CONCLUÍDA!")
    print("="*70)
    print(f"Arquivos gerados em Resultados/:")
    print(f"  • matriz_confusao_completa.npy (1687x1687)")
    print(f"  • matriz_confusao_top15_acuracia.png")
    print(f"  • acuracia_por_classe.png")
    print(f"  • analise_detalhada_modelo.json")
    print("\nPrincipais conclusões:")
    print(f"  • Acurácia geral: {accuracy:.2%}")
    print(f"  • Acurácia média por classe: {np.mean(acc_values):.2%}")
    print(f"  • {np.sum(np.array(acc_values) >= 0.8)} classes com ≥80% de acurácia")
    print("="*70)

if __name__ == "__main__":
    # Configurar matplotlib para usar backend não-interativo
    import matplotlib
    matplotlib.use('Agg')
    
    main()
    