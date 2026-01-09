#!/usr/bin/env python3
"""
PRÉ-PROCESSAMENTO DO DATASET CelebA
Utiliza 25% das imagens do CelebA (como trabalho anterior)

OBJETIVO:
- Processar 25% do CelebA original (50.650 imagens aproximadamente)
- Redimensionar para tamanho escolhido (64x64, 128x128 ou 224x224)
- Aplicar normalização
- Converter para grayscale (opcional)
- Dividir em train/val/test (70%/15%/15%)
"""

import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json
import random
import shutil

def parse_arguments():
    """Configura os parâmetros do pré-processamento"""
    parser = argparse.ArgumentParser(description='Pré-processamento do CelebA - 25% da base')
    
    parser.add_argument('--input_dir', type=str, default='Images/Original_images',
                       help='Diretório com imagens originais do CelebA')
    
    parser.add_argument('--output_dir', type=str, default='Images/Processed',
                       help='Diretório para salvar dados processados')
    
    parser.add_argument('--resize', type=int, default=128,
                       choices=[64, 128, 224],
                       help='Tamanho para redimensionamento')
    
    parser.add_argument('--grayscale', action='store_true',
                       help='Converter para escala de cinza (default: RGB)')
    
    parser.add_argument('--sample_ratio', type=float, default=0.25,
                       help='Percentual da base a ser utilizado (0.25 = 25%)')
    
    parser.add_argument('--max_images', type=int, default=50650,
                       help='Número máximo de imagens a processar (25% de 202.599 ≈ 50.650)')
    
    parser.add_argument('--min_images_per_class', type=int, default=5,
                       help='Mínimo de imagens por pessoa (filtrar classes pequenas)')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Seed para reprodutibilidade')
    
    return parser.parse_args()

def get_image_paths(input_dir, sample_ratio=0.25, max_images=50650, seed=42):
    """
    Obtém lista de caminhos de imagens, amostrando 25% do dataset
    
    CelebA tem estrutura:
        img_align_celeba/
        ├── 000001.jpg
        ├── 000002.jpg
        └── ...
    """
    print(f"\nProcurando imagens em: {input_dir}")
    
    # Lista todos os arquivos JPG
    all_images = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append(os.path.join(root, file))
    
    print(f"Total de imagens encontradas: {len(all_images):,}")
    
    # Amostra aleatória de 25%
    random.seed(seed)
    sample_size = min(int(len(all_images) * sample_ratio), max_images)
    sampled_images = random.sample(all_images, sample_size)
    
    print(f"Amostrando {sample_ratio*100}%: {len(sampled_images):,} imagens")
    
    return sampled_images

def load_and_preprocess_image(image_path, target_size=(128, 128), grayscale=False):
    """Carrega e pré-processa uma única imagem"""
    try:
        # Carrega imagem
        if grayscale:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR → RGB
        
        if img is None:
            print(f"Erro ao carregar: {image_path}")
            return None
        
        # Redimensiona
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        
        # Normaliza para [0, 1]
        img = img.astype('float32') / 255.0
        
        # Adiciona dimensão do canal se for grayscale
        if grayscale:
            img = np.expand_dims(img, axis=-1)  # (H, W) → (H, W, 1)
        
        return img
    
    except Exception as e:
        print(f"Erro ao processar {image_path}: {e}")
        return None

def create_labels_from_filenames(image_paths):
    """
    Cria labels a partir dos nomes dos arquivos.
    No CelebA, cada imagem tem ID único, não há pastas por pessoa.
    Para simulação, vamos agrupar por prefixos.
    """
    # Extrai IDs das imagens (ex: "000001" de "000001.jpg")
    image_ids = []
    for path in image_paths:
        filename = os.path.basename(path)
        # Remove extensão e pega os primeiros 6 dígitos
        img_id = os.path.splitext(filename)[0][:6]
        image_ids.append(img_id)
    
    # Cria labels numéricas (cada ID único vira uma classe)
    unique_ids = list(set(image_ids))
    label_map = {id: idx for idx, id in enumerate(unique_ids)}
    labels = [label_map[img_id] for img_id in image_ids]
    
    return np.array(labels), unique_ids, label_map

def save_processed_data(X_train, y_train, X_val, y_val, X_test, y_test, 
                       class_names, output_dir, grayscale=False):
    """Salva os dados processados em formato numpy"""
    
    # Cria diretórios
    os.makedirs(output_dir, exist_ok=True)
    
    # Salva arrays numpy
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    # Salva metadados
    metadata = {
        'num_classes': len(class_names),
        'class_names': class_names,
        'input_shape': X_train.shape[1:],
        'grayscale': grayscale,
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'total_images': len(X_train) + len(X_val) + len(X_test)
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nDados salvos em: {output_dir}")
    print(f"Formato das imagens: {X_train.shape[1:]}")
    print(f"Número de classes: {len(class_names)}")
    print(f"Imagens de treino: {len(X_train):,}")
    print(f"Imagens de validação: {len(X_val):,}")
    print(f"Imagens de teste: {len(X_test):,}")

def main():
    """Função principal de pré-processamento"""
    args = parse_arguments()
    
    print("="*60)
    print("PRÉ-PROCESSAMENTO DO CelebA - 25% DA BASE")
    print("="*60)
    print(f"Diretório de entrada: {args.input_dir}")
    print(f"Diretório de saída: {args.output_dir}")
    print(f"Tamanho de redimensionamento: {args.resize}x{args.resize}")
    print(f"Grayscale: {args.grayscale}")
    print(f"Percentual amostrado: {args.sample_ratio*100}%")
    print(f"Seed para reprodutibilidade: {args.seed}")
    print("="*60)
    
    # 1. Obtém lista de imagens (25% do dataset)
    image_paths = get_image_paths(
        args.input_dir, 
        sample_ratio=args.sample_ratio,
        max_images=args.max_images,
        seed=args.seed
    )
    
    if not image_paths:
        print("Nenhuma imagem encontrada!")
        return
    
    # 2. Cria labels a partir dos nomes dos arquivos
    print("\nCriando labels...")
    labels, class_names, label_map = create_labels_from_filenames(image_paths)
    
    print(f"Número de classes (IDs únicos): {len(class_names)}")
    print(f"Número total de imagens: {len(image_paths):,}")
    
    # 3. Carrega e pré-processa imagens
    print(f"\nCarregando e pré-processando {len(image_paths):,} imagens...")
    print(f"Tamanho alvo: {args.resize}x{args.resize}")
    print(f"Grayscale: {args.grayscale}")
    
    processed_images = []
    valid_indices = []
    
    with tqdm(total=len(image_paths), desc="Processando imagens") as pbar:
        for i, img_path in enumerate(image_paths):
            img = load_and_preprocess_image(
                img_path, 
                target_size=(args.resize, args.resize),
                grayscale=args.grayscale
            )
            
            if img is not None:
                processed_images.append(img)
                valid_indices.append(i)
            pbar.update(1)
    
    # Filtra labels correspondentes às imagens válidas
    valid_labels = labels[valid_indices]
    
    if not processed_images:
        print("Nenhuma imagem foi processada com sucesso!")
        return
    
    X = np.array(processed_images)
    y = valid_labels
    
    print(f"\nDataset processado:")
    print(f"  Formato de X: {X.shape}")
    print(f"  Formato de y: {y.shape}")
    print(f"  Valores de pixel: [{X.min():.3f}, {X.max():.3f}]")
    
    # 4. Divide em train/val/test (70%/15%/15%)
    print("\nDividindo em conjuntos train/val/test...")
    
    # Primeira divisão: treino (70%) vs temporário (30%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=args.seed
    )
    
    # Segunda divisão: treino (70%) vs validação (15% do original)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=args.seed
    )  # 0.1765 ≈ 15/85 para obter 15% do total
    
    print(f"Divisão concluída:")
    print(f"  Treino: {len(X_train):,} imagens ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Validação: {len(X_val):,} imagens ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Teste: {len(X_test):,} imagens ({len(X_test)/len(X)*100:.1f}%)")
    
    # 5. Salva os dados processados
    print("\nSalvando dados processados...")
    save_processed_data(
        X_train, y_train, X_val, y_val, X_test, y_test,
        class_names, args.output_dir, args.grayscale
    )
    
    # 6. Salva algumas imagens de exemplo para visualização
    save_sample_images(X_train, y_train, class_names, args.output_dir, args.grayscale)
    
    print("\n" + "="*60)
    print("PRÉ-PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
    print("="*60)

def save_sample_images(X, y, class_names, output_dir, grayscale=False, num_samples=16):
    """Salva algumas imagens de exemplo para visualização"""
    import matplotlib.pyplot as plt
    
    samples_dir = os.path.join(output_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    
    # Seleciona amostras aleatórias
    indices = np.random.choice(len(X), min(num_samples, len(X)), replace=False)
    
    plt.figure(figsize=(12, 12))
    for i, idx in enumerate(indices):
        plt.subplot(4, 4, i + 1)
        
        img = X[idx]
        if grayscale and img.shape[-1] == 1:
            img = img.squeeze()  # Remove dimensão do canal
        
        plt.imshow(img, cmap='gray' if grayscale else None)
        plt.title(f"Classe: {class_names[y[idx]]}", fontsize=8)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(samples_dir, 'sample_images.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nAmostras salvas em: {samples_dir}/sample_images.png")

if __name__ == "__main__":
    main()