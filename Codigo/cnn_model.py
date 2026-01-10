#!/usr/bin/env python3
"""
CNN PARA RECONHECIMENTO FACIAL NO CelebA
Versão simplificada e robusta
"""

import os
import sys
import numpy as np
import json
import time
import argparse
from datetime import datetime

# Importações obrigatórias
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers

# Importações opcionais (com fallback)
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Aviso: matplotlib não disponível")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("Aviso: seaborn não disponível")

try:
    from sklearn.metrics import confusion_matrix, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Aviso: scikit-learn não disponível")

def parse_args():
    """Configura argumentos do modelo"""
    parser = argparse.ArgumentParser(description='Treinar CNN para reconhecimento facial')
    
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Tamanho do batch')
    parser.add_argument('--epochs', type=int, default=10,  # Comece com poucas épocas
                       help='Número de épocas')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Taxa de aprendizado')
    parser.add_argument('--model_name', type=str, default='cnn_facial',
                       help='Nome do modelo')
    parser.add_argument('--simple', action='store_true',
                       help='Usar arquitetura mais simples')
    parser.add_argument('--test_only', action='store_true',
                       help='Apenas testar modelo existente')
    
    return parser.parse_args()

def load_data():
    """Carrega os dados pré-processados de forma eficiente"""
    print("\n" + "="*60)
    print("[1/4] CARREGANDO DADOS")
    print("="*60)
    
    base_path = os.path.join('..', 'Images', 'Processed')
    
    # Verifica se os dados existem
    if not os.path.exists(base_path):
        print("❌ ERRO: Dados não encontrados!")
        print("Execute primeiro: python preprocess.py")
        sys.exit(1)
    
    print(f"Carregando de: {base_path}")
    
    try:
        # Carrega metadados primeiro
        with open(os.path.join(base_path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        print(f"Dataset: {metadata['estatisticas']['total_imagens']:,} imagens")
        print(f"Classes: {metadata['estatisticas']['classes']}")
        print(f"Formato: {metadata['formato_imagens']}")
        
        # Carrega em ordem para gerenciar memória
        print("\nCarregando arrays...")
        
        X_train = np.load(os.path.join(base_path, 'X_train.npy'))
        y_train = np.load(os.path.join(base_path, 'y_train.npy'))
        
        X_val = np.load(os.path.join(base_path, 'X_val.npy'))
        y_val = np.load(os.path.join(base_path, 'y_val.npy'))
        
        X_test = np.load(os.path.join(base_path, 'X_test.npy'))
        y_test = np.load(os.path.join(base_path, 'y_test.npy'))
        
        num_classes = metadata['estatisticas']['classes']
        input_shape = tuple(metadata['formato_imagens'])
        
        print(f"\n✓ Dados carregados:")
        print(f"  Treino:   {X_train.shape[0]:,} imagens")
        print(f"  Validação: {X_val.shape[0]:,} imagens")
        print(f"  Teste:    {X_test.shape[0]:,} imagens")
        print(f"  Total:    {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]:,} imagens")
        
        return X_train, y_train, X_val, y_val, X_test, y_test, num_classes, input_shape, metadata
        
    except Exception as e:
        print(f"❌ Erro ao carregar: {str(e)}")
        sys.exit(1)

def build_model_simple(input_shape, num_classes):
    """Modelo CNN simples e eficiente"""
    print(f"\n[2/4] CONSTRUINDO MODELO SIMPLES")
    print("="*60)
    
    model = keras.Sequential([
        # Entrada
        keras.Input(shape=input_shape),
        
        # Bloco convolucional 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Bloco convolucional 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Bloco convolucional 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Classificador
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def build_model_advanced(input_shape, num_classes):
    """Modelo CNN mais avançado"""
    print(f"\n[2/4] CONSTRUINDO MODELO AVANÇADO")
    print("="*60)
    
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        
        # Data augmentation
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        
        # Bloco 1
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Bloco 2
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Bloco 3
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Classificador
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, args):
    """Treina o modelo"""
    print(f"\n[3/4] TREINANDO MODELO")
    print("="*60)
    
    # Compilar
    model.compile(
        optimizer=optimizers.Adam(learning_rate=args.learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5_acc')]
    )
    
    print(f"Configuração:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Épocas: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Otimizador: Adam")
    print(f"  Métricas: Accuracy, Top-5 Accuracy")
    
    # Callbacks
    callbacks_list = []
    
    # Checkpoint
    checkpoint_path = os.path.join('..', 'Resultados', f'{args.model_name}_best.h5')
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    callbacks_list.append(
        callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    )
    
    # Early stopping
    callbacks_list.append(
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    )
    
    # Reduce LR
    callbacks_list.append(
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    )
    
    # TensorBoard
    log_dir = os.path.join('..', 'Resultados', 'logs', args.model_name)
    callbacks_list.append(
        callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    )
    
    # Treinar
    print(f"\nIniciando treinamento...")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  Classes: {len(np.unique(y_train))}")
    
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks_list,
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    print(f"\n✓ Treinamento concluído em {training_time/60:.2f} minutos")
    print(f"  Melhor val_accuracy: {max(history.history['val_accuracy']):.4f}")
    
    return history, training_time

def evaluate_model(model, X_test, y_test, num_classes, args):
    """Avalia o modelo"""
    print(f"\n[4/4] AVALIANDO MODELO")
    print("="*60)
    
    # Avaliação básica
    print("Avaliando no conjunto de teste...")
    test_loss, test_acc, test_top5 = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\nResultados no teste:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Top-5 Accuracy: {test_top5:.4f} ({test_top5*100:.2f}%)")
    
    # Previsões (amostra para economia de memória)
    if num_classes > 100:
        print(f"\nNota: Many classes ({num_classes}), skipping detailed predictions")
        return test_loss, test_acc, test_top5
    
    # Para datasets menores, fazer previsões detalhadas
    print(f"\nGerando previsões...")
    y_pred = model.predict(X_test, batch_size=args.batch_size, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Classification report para algumas classes
    if SKLEARN_AVAILABLE and num_classes <= 20:
        print("\nClassification Report (primeiras 5 classes):")
        print(classification_report(
            y_test, y_pred_classes,
            target_names=[f'Class_{i}' for i in range(min(5, num_classes))],
            zero_division=0
        ))
    
    return test_loss, test_acc, test_top5

def save_results(model, history, test_results, training_time, metadata, args):
    """Salva resultados"""
    print(f"\n" + "="*60)
    print("SALVANDO RESULTADOS")
    print("="*60)
    
    results_dir = os.path.join('..', 'Resultados')
    os.makedirs(results_dir, exist_ok=True)
    
    # Salva modelo final
    model_path = os.path.join(results_dir, f'{args.model_name}_final.h5')
    model.save(model_path)
    print(f"✓ Modelo salvo: {model_path}")
    
    # Salva histórico de treinamento
    history_path = os.path.join(results_dir, f'{args.model_name}_history.npy')
    np.save(history_path, history.history)
    print(f"✓ Histórico salvo: {history_path}")
    
    # Salva métricas em JSON
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'model_name': args.model_name,
        'dataset': {
            'total_images': metadata['estatisticas']['total_imagens'],
            'classes': metadata['estatisticas']['classes'],
            'train_size': metadata['estatisticas']['treino'],
            'val_size': metadata['estatisticas']['validacao'],
            'test_size': metadata['estatisticas']['teste']
        },
        'training': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'training_time_minutes': training_time / 60,
            'final_epoch': len(history.history['loss'])
        },
        'results': {
            'test_loss': float(test_results[0]),
            'test_accuracy': float(test_results[1]),
            'test_top5_accuracy': float(test_results[2])
        },
        'best_validation': {
            'val_accuracy': float(max(history.history['val_accuracy'])),
            'val_loss': float(min(history.history['val_loss']))
        }
    }
    
    metrics_path = os.path.join(results_dir, f'{args.model_name}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Métricas salvas: {metrics_path}")
    
    # Plota gráficos se matplotlib disponível
    if MATPLOTLIB_AVAILABLE:
        try:
            plt.figure(figsize=(12, 4))
            
            # Accuracy
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Train')
            plt.plot(history.history['val_accuracy'], label='Validation')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            
            # Loss
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Train')
            plt.plot(history.history['val_loss'], label='Validation')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plot_path = os.path.join(results_dir, f'{args.model_name}_training.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Gráfico salvo: {plot_path}")
            
        except Exception as e:
            print(f"⚠  Não foi possível salvar gráficos: {str(e)}")
    
    return metrics

def main():
    """Função principal"""
    args = parse_args()
    
    print("="*70)
    print("CNN PARA RECONHECIMENTO FACIAL - CelebA")
    print("="*70)
    print(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    try:
        # 1. Carrega dados
        X_train, y_train, X_val, y_val, X_test, y_test, num_classes, input_shape, metadata = load_data()
        
        # 2. Constrói modelo
        if args.simple or num_classes > 500:
            model = build_model_simple(input_shape, num_classes)
        else:
            model = build_model_advanced(input_shape, num_classes)
        
        # Resumo do modelo
        model.summary()
        
        # 3. Treina modelo
        history, training_time = train_model(model, X_train, y_train, X_val, y_val, args)
        
        # 4. Avalia modelo
        test_results = evaluate_model(model, X_test, y_test, num_classes, args)
        
        # 5. Salva resultados
        metrics = save_results(model, history, test_results, training_time, metadata, args)
        
        # 6. Resumo final
        print("\n" + "="*70)
        print("TREINAMENTO CONCLUÍDO!")
        print("="*70)
        print(f"Modelo: {args.model_name}")
        print(f"Classes: {num_classes}")
        print(f"Test Accuracy: {metrics['results']['test_accuracy']:.4f} ({metrics['results']['test_accuracy']*100:.2f}%)")
        print(f"Test Top-5 Accuracy: {metrics['results']['test_top5_accuracy']:.4f} ({metrics['results']['test_top5_accuracy']*100:.2f}%)")
        print(f"Tempo total: {metrics['training']['training_time_minutes']:.2f} minutos")
        print("="*70)
        print(f"\nArquivos em Resultados/:")
        print(f"  {args.model_name}_final.h5 - Modelo final")
        print(f"  {args.model_name}_best.h5 - Melhor modelo")
        print(f"  {args.model_name}_metrics.json - Métricas")
        print(f"  {args.model_name}_training.png - Gráficos (se disponível)")
        print(f"  logs/ - Logs do TensorBoard")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ ERRO: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()