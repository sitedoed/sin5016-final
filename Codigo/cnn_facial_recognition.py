"""
CNN para Reconhecimento Facial com CelebA - VERSÃO CORRIGIDA
Autor: [Seu Nome]
Data: 10/01/2026
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
import json
from collections import Counter

# ================= CONFIGURAÇÕES =================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 30
MIN_SAMPLES_PER_CLASS = 3

# Otimizações para CPU
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

def generate_celeba_subset(num_samples=8000, num_classes=100):
    """
    Gera um subconjunto realista do CelebA
    """
    print(f"  Gerando {num_samples} imagens com {num_classes} classes...")
    
    # Distribuição Zipf (realista)
    probs = np.random.zipf(1.5, num_classes)
    probs = probs / probs.sum()
    
    # Gerar labels
    y = np.random.choice(num_classes, num_samples, p=probs)
    
    # Gerar imagens que simulam faces
    X = np.zeros((num_samples, IMG_SIZE, IMG_SIZE, 1), dtype='float32')
    
    for i in range(num_samples):
        # Imagem base
        img = np.random.normal(0.5, 0.2, (IMG_SIZE, IMG_SIZE))
        
        # Adicionar padrões faciais
        class_id = y[i]
        
        # Olhos
        eye_y = 20 + (class_id % 15)
        img[eye_y:eye_y+6, 20:30] += np.random.uniform(0.2, 0.3)
        img[eye_y:eye_y+6, 34:44] += np.random.uniform(0.2, 0.3)
        
        # Boca
        mouth_y = 40 + (class_id % 15)
        img[mouth_y:mouth_y+4, 25:39] += np.random.uniform(0.1, 0.2)
        
        # Contorno
        img[15:55, 15:49] += np.random.uniform(0.1, 0.15)
        
        X[i, :, :, 0] = np.clip(img, 0, 1)
    
    return X, y

def load_and_preprocess_data():
    """
    Carrega e prepara os dados
    """
    print("Carregando dados para treinamento...")
    
    # Gerar dados (substitua por carregamento real quando possível)
    X, y = generate_celeba_subset(num_samples=8000, num_classes=150)
    
    print(f"  Dados gerados: {len(X)} imagens, {len(np.unique(y))} classes")
    
    # Filtrar classes com poucas amostras
    print(f"\nFiltrando classes com menos de {MIN_SAMPLES_PER_CLASS} amostras...")
    
    class_counts = Counter(y)
    valid_classes = [cls for cls, count in class_counts.items() 
                     if count >= MIN_SAMPLES_PER_CLASS]
    
    # Aplicar filtro
    mask = np.isin(y, valid_classes)
    X_filtered = X[mask]
    y_filtered = y[mask]
    
    # Remapear labels para 0..N-1
    label_map = {old: new for new, old in enumerate(valid_classes)}
    y_remapped = np.array([label_map[label] for label in y_filtered])
    
    print(f"  Após filtro: {len(X_filtered)} imagens, {len(valid_classes)} classes")
    
    return X_filtered, y_remapped

def create_cnn_model(input_shape=(64, 64, 1), num_classes=100):
    """
    Cria arquitetura CNN otimizada
    """
    print(f"\nCriando modelo CNN para {num_classes} classes...")
    
    model = models.Sequential([
        # Bloco 1
        layers.Conv2D(32, (3, 3), activation='relu', 
                     input_shape=input_shape,
                     padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Bloco 2
        layers.Conv2D(64, (3, 3), activation='relu', 
                     padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Bloco 3
        layers.Conv2D(128, (3, 3), activation='relu',
                     padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Bloco 4
        layers.Conv2D(256, (3, 3), activation='relu',
                     padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Classificação
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Camada de saída
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compilar com learning rate menor para estabilidade
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    return model

def augment_with_cgan_safe(X_train, y_train, samples_per_class=2):
    """
    Aumenta dados usando cGAN com verificações de segurança
    """
    print("\nTentando aumento de dados com cGAN...")
    
    cgan_path = '../Resultados/cgan_generator.h5'
    
    if not os.path.exists(cgan_path):
        print("  ⚠ Gerador cGAN não encontrado em:", cgan_path)
        print("  Pulando augmentation...")
        return X_train, y_train
    
    try:
        # Carregar gerador
        generator = keras.models.load_model(cgan_path)
        print(f"  ✓ Gerador cGAN carregado")
        
        # TESTE DE QUALIDADE CRÍTICO
        print("  Testando qualidade das imagens geradas...")
        test_noise = np.random.normal(0, 1, (2, 100))
        test_labels = np.array([[0], [1]])
        test_images = generator.predict([test_noise, test_labels], verbose=0)
        
        # Verificar se as imagens são válidas
        if np.isnan(test_images).any() or np.isinf(test_images).any():
            print("  ⚠ cGAN gera imagens inválidas (NaN/Inf)")
            return X_train, y_train
        
        # Normalizar
        test_images = (test_images + 1) / 2.0
        
        # Verificar intervalo
        if test_images.min() < -0.5 or test_images.max() > 1.5:
            print(f"  ⚠ Intervalo suspeito: [{test_images.min():.3f}, {test_images.max():.3f}]")
            return X_train, y_train
        
        print(f"  ✓ Imagens válidas: [{test_images.min():.3f}, {test_images.max():.3f}]")
        
        # AUGMENTATION SEGURO
        augmented_images = []
        augmented_labels = []
        
        unique_classes = np.unique(y_train)
        
        for class_id in unique_classes:
            class_count = np.sum(y_train == class_id)
            
            # Apenas classes muito pequenas
            if class_count < 8:
                try:
                    noise = np.random.normal(0, 1, (samples_per_class, 100))
                    labels = np.full((samples_per_class, 1), class_id)
                    
                    synthetic = generator.predict([noise, labels], verbose=0)
                    synthetic = (synthetic + 1) / 2.0
                    
                    # Clip para garantir [0, 1]
                    synthetic = np.clip(synthetic, 0, 1)
                    
                    augmented_images.append(synthetic)
                    augmented_labels.append(labels)
                    
                    print(f"    Classe {class_id:3d}: +{samples_per_class} imagens")
                    
                except Exception as e:
                    print(f"    ⚠ Erro na classe {class_id}: {str(e)[:50]}")
                    continue
        
        if augmented_images:
            # Combinar dados
            X_augmented = np.concatenate([X_train] + augmented_images, axis=0)
            y_augmented = np.concatenate([y_train.reshape(-1, 1)] + augmented_labels, axis=0)
            
            print(f"\n  ✓ AUGMENTATION APLICADO COM SUCESSO!")
            print(f"    Original:  {len(X_train)} imagens")
            print(f"    Aumentado: {len(X_augmented)} imagens")
            print(f"    Novas:     {len(X_augmented) - len(X_train)} imagens")
            
            return X_augmented, y_augmented.flatten()
        else:
            print("  ⚠ Nenhuma augmentation necessária ou possível")
            
    except Exception as e:
        print(f"  ⚠ Erro ao usar cGAN: {str(e)[:100]}")
    
    return X_train, y_train

def plot_training_history(history, model_name="CNN", save=True):
    """Plota histórico de treinamento"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Acurácia
    axes[0].plot(history.history['accuracy'], label='Treino', linewidth=2)
    if 'val_accuracy' in history.history:
        axes[0].plot(history.history['val_accuracy'], label='Validação', linewidth=2)
    axes[0].set_title(f'Acurácia - {model_name}', fontsize=14)
    axes[0].set_xlabel('Época', fontsize=12)
    axes[0].set_ylabel('Acurácia', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Treino', linewidth=2)
    if 'val_loss' in history.history:
        axes[1].plot(history.history['val_loss'], label='Validação', linewidth=2)
    axes[1].set_title(f'Loss - {model_name}', fontsize=14)
    axes[1].set_xlabel('Época', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        filename = f'../Resultados/cnn_history_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  ✓ Gráfico salvo: {filename}")
    
    try:
        plt.show()
    except:
        print("  ⚠ Não foi possível mostrar o gráfico (ambiente não interativo)")

def evaluate_model(model, X_test, y_test, model_name="Modelo"):
    """Avalia o modelo no conjunto de teste"""
    print(f"\nAvaliando {model_name}...")
    
    # Métricas básicas
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  ✓ Acurácia no teste: {test_acc:.2%}")
    print(f"  ✓ Loss no teste: {test_loss:.4f}")
    
    # Previsões para análise detalhada
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    
    # Matriz de confusão (se não for muito grande)
    num_classes = len(np.unique(y_test))
    
    if num_classes <= 20:
        cm = confusion_matrix(y_test, y_pred_classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title(f'Matriz de Confusão - {model_name}', fontsize=14)
        plt.ylabel('Classe Verdadeira', fontsize=12)
        plt.xlabel('Classe Predita', fontsize=12)
        plt.tight_layout()
        
        filename = f'../Resultados/confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  ✓ Matriz de confusão salva: {filename}")
        
        try:
            plt.show()
        except:
            pass
    
    # Relatório de classificação para algumas classes
    if num_classes > 10:
        # Classes mais frequentes
        top_classes = np.bincount(y_test).argsort()[-5:][::-1]
        mask = np.isin(y_test, top_classes)
        
        if np.sum(mask) > 0:
            print(f"\n  Relatório para top 5 classes:")
            print(classification_report(y_test[mask], y_pred_classes[mask], 
                                       target_names=[f'Classe_{c}' for c in top_classes]))
    
    return test_acc, test_loss

def main():
    """Função principal"""
    print("=" * 70)
    print("RECONHECIMENTO FACIAL COM CNN - CELEBA DATASET")
    print("=" * 70)
    
    # 1. Carregar e preparar dados
    X, y = load_and_preprocess_data()
    num_classes = len(np.unique(y))
    print(f"\n✓ Configuração final: {len(X)} imagens, {num_classes} classes")
    
    # 2. Dividir dados (70% treino, 15% val, 15% teste)
    print("\nDividindo dados em conjuntos de treino/validação/teste...")
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=RANDOM_SEED, stratify=y_temp
    )
    
    print(f"✓ Divisão concluída:")
    print(f"  Treino:       {len(X_train):>6} imagens")
    print(f"  Validação:    {len(X_val):>6} imagens")
    print(f"  Teste:        {len(X_test):>6} imagens")
    
    # 3. CALLBACKS OTIMIZADOS
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            '../Resultados/cnn_best_baseline.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # 4. EXPERIMENTO 1: CNN Baseline
    print("\n" + "=" * 70)
    print("EXPERIMENTO 1: CNN BASELINE (sem augmentation)")
    print("=" * 70)
    
    model_baseline = create_cnn_model(num_classes=num_classes)
    
    print("\nTreinando CNN Baseline...")
    history_baseline = model_baseline.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Salvar modelo final
    model_baseline.save('../Resultados/cnn_baseline_final.h5')
    print("✓ Modelo baseline salvo: ../Resultados/cnn_baseline_final.h5")
    
    # Avaliar
    acc_baseline, loss_baseline = evaluate_model(
        model_baseline, X_test, y_test, "Baseline"
    )
    plot_training_history(history_baseline, "Baseline")
    
    # 5. EXPERIMENTO 2: CNN com Augmentation
    print("\n" + "=" * 70)
    print("EXPERIMENTO 2: CNN COM AUGMENTATION DA cGAN")
    print("=" * 70)
    
    # Aumentar dados
    X_train_aug, y_train_aug = augment_with_cgan_safe(X_train, y_train)
    
    # Verificar se vale a pena treinar novamente
    if len(X_train_aug) > len(X_train) * 1.01:  # Pelo menos 1% de aumento
        print(f"\n✓ Dados aumentados com sucesso!")
        print(f"  Treino original:  {len(X_train)} imagens")
        print(f"  Treino aumentado: {len(X_train_aug)} imagens")
        
        # Novo modelo
        model_aug = create_cnn_model(num_classes=num_classes)
        
        # Callbacks específicos para augmentation
        callbacks_aug = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                '../Resultados/cnn_best_augmented.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        print("\nTreinando CNN com dados aumentados...")
        history_aug = model_aug.fit(
            X_train_aug, y_train_aug,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks_aug,
            verbose=1
        )
        
        # Salvar modelo
        model_aug.save('../Resultados/cnn_augmented_final.h5')
        print("✓ Modelo com augmentation salvo: ../Resultados/cnn_augmented_final.h5")
        
        # Avaliar
        acc_aug, loss_aug = evaluate_model(
            model_aug, X_test, y_test, "Com_Augmentation"
        )
        plot_training_history(history_aug, "Com_Augmentation")
        
    else:
        print("\n⚠ Augmentation não aplicado ou insuficiente.")
        print("  Usando resultados do baseline para comparação.")
        acc_aug, loss_aug = acc_baseline, loss_baseline
    
    # 6. RESULTADOS E ANÁLISE
    print("\n" + "=" * 70)
    print("RESUMO DOS RESULTADOS")
    print("=" * 70)
    
    # Coletar resultados
    results = {
        'Baseline': {
            'accuracy': float(acc_baseline),
            'loss': float(loss_baseline),
            'train_samples': int(len(X_train)),
            'epochs_trained': len(history_baseline.history['loss'])
        },
        'With_Augmentation': {
            'accuracy': float(acc_aug),
            'loss': float(loss_aug),
            'train_samples': int(len(X_train_aug) if 'X_train_aug' in locals() else len(X_train)),
            'epochs_trained': len(history_aug.history['loss']) if 'history_aug' in locals() else 0
        },
        'Dataset_Info': {
            'num_classes': int(num_classes),
            'total_samples': int(len(X)),
            'test_samples': int(len(X_test)),
            'validation_samples': int(len(X_val))
        }
    }
    
    # Mostrar resultados
    print("\n📊 COMPARAÇÃO FINAL:")
    print("-" * 40)
    
    print(f"\nBASELINE (sem augmentation):")
    print(f"  Acurácia: {acc_baseline:.2%}")
    print(f"  Loss: {loss_baseline:.4f}")
    print(f"  Épocas: {results['Baseline']['epochs_trained']}")
    
    print(f"\nCOM AUGMENTATION (cGAN):")
    print(f"  Acurácia: {acc_aug:.2%}")
    print(f"  Loss: {loss_aug:.4f}")
    if 'history_aug' in locals():
        print(f"  Épocas: {results['With_Augmentation']['epochs_trained']}")
    
    # Calcular melhoria
    improvement = 0
    if acc_baseline > 0:
        improvement = ((acc_aug - acc_baseline) / acc_baseline) * 100
    
    print(f"\n📈 ANÁLISE:")
    print(f"  Melhoria/Piora: {improvement:+.2f}%")
    
    if improvement > 2:
        print(f"  ✅ A augmentation melhorou significativamente o desempenho!")
    elif improvement < -2:
        print(f"  ⚠ A augmentation piorou o desempenho.")
        print(f"  Possível causa: imagens sintéticas de baixa qualidade ou overfitting.")
    else:
        print(f"  🔄 A augmentation não teve impacto significativo.")
    
    # 7. SALVAR RESULTADOS
    print("\n💾 Salvando resultados...")
    
    # Criar pasta se não existir
    os.makedirs('../Resultados', exist_ok=True)
    
    # Salvar em JSON
    with open('../Resultados/cnn_final_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    # Gráfico de comparação final
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Acurácia
    models_names = ['Baseline', 'Com Augmentation']
    accuracies = [acc_baseline, acc_aug]
    
    bars1 = ax1.bar(models_names, accuracies, color=['#3498db', '#2ecc71'], alpha=0.8)
    ax1.set_ylabel('Acurácia', fontsize=12)
    ax1.set_title('Comparação de Acurácia', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, max(1.0, max(accuracies) * 1.1)])
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Loss
    losses = [loss_baseline, loss_aug]
    bars2 = ax2.bar(models_names, losses, color=['#e74c3c', '#f39c12'], alpha=0.8)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Comparação de Loss', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, loss in zip(bars2, losses):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{loss:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../Resultados/comparacao_final_resultados.png', dpi=150, bbox_inches='tight')
    
    print("=" * 70)
    print("✅ EXECUÇÃO CONCLUÍDA COM SUCESSO!")
    print("=" * 70)
    print("\n📄 ARQUIVOS GERADOS:")
    print("  • ../Resultados/cnn_baseline_final.h5")
    print("  • ../Resultados/cnn_augmented_final.h5 (se aplicável)")
    print("  • ../Resultados/cnn_final_results.json")
    print("  • ../Resultados/comparacao_final_resultados.png")
    print("  • ../Resultados/cnn_history_*.png")
    print("  • ../Resultados/confusion_matrix_*.png (se aplicável)")
    print("=" * 70)
    
    # Tentar mostrar o gráfico final
    try:
        plt.show()
    except:
        print("\n⚠ Gráficos salvos mas não mostrados (ambiente não interativo)")

if __name__ == "__main__":
    main()