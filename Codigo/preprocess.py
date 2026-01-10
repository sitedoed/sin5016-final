#!/usr/bin/env python3
"""
PRÉ-PROCESSAMENTO OTIMIZADO - CORRIGIDO
Remove classes com menos de 2 imagens ANTES da divisão
"""

import os
import cv2
import numpy as np
import json
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse
from collections import defaultdict, Counter
import sys

def parse_args():
    """Configura argumentos otimizados"""
    parser = argparse.ArgumentParser(description='Pré-processar CelebA (corrigido)')
    
    parser.add_argument('--percentual', type=float, default=0.10,
                       help='Percentual da base (0.10 = 10% para teste)')
    parser.add_argument('--min_img', type=int, default=10,
                       help='Mínimo de imagens por pessoa')
    parser.add_argument('--min_final', type=int, default=2,
                       help='Mínimo final por classe (default: 2)')
    parser.add_argument('--tamanho', type=int, default=64,
                       choices=[64, 128],
                       help='Tamanho da imagem')
    parser.add_argument('--cinza', action='store_true',
                       help='Converter para escala de cinza')
    parser.add_argument('--batch_size', type=int, default=1000,
                       help='Processar em batches')
    parser.add_argument('--seed', type=int, default=42,
                       help='Semente aleatória')
    
    return parser.parse_args()

def carregar_identidades():
    """Carrega identidades"""
    print(f"\n[1/6] Carregando identidades...")
    
    caminho = os.path.join('..', 'Docs', 'identity_CelebA.txt')
    if not os.path.exists(caminho):
        caminho = 'identity_CelebA.txt'
    
    pessoa_para_imagens = defaultdict(list)
    total_linhas = 0
    
    with open(caminho, 'r') as f:
        for i, linha in enumerate(f):
            if i == 0 and linha.strip().isdigit():
                continue
            
            partes = linha.strip().split()
            if len(partes) >= 2:
                nome_img = partes[0]
                id_pessoa = int(partes[1])
                pessoa_para_imagens[id_pessoa].append(nome_img)
                total_linhas += 1
    
    print(f"  Total imagens: {total_linhas:,}")
    print(f"  Pessoas únicas: {len(pessoa_para_imagens):,}")
    
    return pessoa_para_imagens, total_linhas

def selecionar_amostra_corrigida(pessoa_para_imagens, total_imagens, percentual=0.10, min_img=10, min_final=2, seed=42):
    """Seleciona amostra garantindo mínimo final"""
    print(f"\n[2/6] Selecionando {percentual*100:.0f}%...")
    
    imagens_alvo = int(total_imagens * percentual)
    print(f"  Alvo: {imagens_alvo:,} imagens")
    print(f"  Mínimo inicial: {min_img}, Mínimo final: {min_final}")
    
    # Filtra pessoas com mínimo de imagens
    pessoas_validas = [(pid, imgs) for pid, imgs in pessoa_para_imagens.items() 
                      if len(imgs) >= min_img]
    pessoas_validas.sort(key=lambda x: len(x[1]), reverse=True)
    
    print(f"  Pessoas com ≥{min_img} imagens: {len(pessoas_validas):,}")
    
    random.seed(seed)
    dataset = []
    mapeamento = {}
    contagem_classes = Counter()
    
    # FASE 1: Coleta imagens mantendo contagem
    for idx, (pessoa_id, imagens) in enumerate(pessoas_validas):
        if len(dataset) >= imagens_alvo:
            break
            
        mapeamento[pessoa_id] = idx
        
        # Limita a 100 imagens por pessoa máximo
        if len(imagens) > 100:
            imagens = random.sample(imagens, 100)
        
        for nome_img in imagens:
            dataset.append({
                'imagem': nome_img,
                'pessoa_id': pessoa_id,
                'classe': idx
            })
            contagem_classes[idx] += 1
            
            if len(dataset) >= imagens_alvo:
                break
    
    print(f"  Selecionado inicial: {len(dataset):,} imagens, {len(mapeamento)} pessoas")
    
    # FASE 2: Remove classes com menos de min_final imagens
    classes_remover = [classe for classe, count in contagem_classes.items() 
                      if count < min_final]
    
    if classes_remover:
        print(f"  Removendo {len(classes_remover)} classes com <{min_final} imagens...")
        
        # Remove essas classes do dataset
        dataset = [item for item in dataset if item['classe'] not in classes_remover]
        
        # Re-mapeia classes
        classes_unicas = sorted(set(item['classe'] for item in dataset))
        novo_mapeamento = {}
        novo_mapeamento_inverso = {}
        
        for nova_classe, classe_original in enumerate(classes_unicas):
            novo_mapeamento[classe_original] = nova_classe
            # Encontra pessoa_id original para esta classe
            for item in dataset:
                if item['classe'] == classe_original:
                    pessoa_id_original = item['pessoa_id']
                    novo_mapeamento_inverso[pessoa_id_original] = nova_classe
                    break
        
        # Atualiza classes no dataset
        for item in dataset:
            item['classe'] = novo_mapeamento[item['classe']]
        
        mapeamento = novo_mapeamento_inverso
    
    print(f"  Após filtro: {len(dataset):,} imagens, {len(set(m['classe'] for m in dataset))} pessoas")
    print(f"  Imagens por classe: min={min(Counter(m['classe'] for m in dataset).values())}, "
          f"max={max(Counter(m['classe'] for m in dataset).values())}")
    
    return dataset, mapeamento

def processar_batches(dataset, tamanho, cinza, batch_size=1000):
    """Processa em batches"""
    print(f"\n[3/6] Processando em batches de {batch_size}...")
    print(f"  Tamanho: {tamanho}x{tamanho}, Cinza: {cinza}")
    
    dir_imagens = os.path.join('..', 'Images', 'Original_images', 'img_align_celeba')
    dir_temp = os.path.join('..', 'Images', 'Processed', 'temp')
    os.makedirs(dir_temp, exist_ok=True)
    
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Processando"):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(dataset))
        batch_data = dataset[start:end]
        
        X_batch = []
        y_batch = []
        
        for item in batch_data:
            caminho = os.path.join(dir_imagens, item['imagem'])
            
            try:
                if cinza:
                    img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (tamanho, tamanho))
                        img = np.expand_dims(img, axis=-1)
                else:
                    img = cv2.imread(caminho)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (tamanho, tamanho))
                
                if img is not None:
                    img = img.astype('float32') / 255.0
                    X_batch.append(img)
                    y_batch.append(item['classe'])
                    
            except Exception:
                continue
        
        if X_batch:
            batch_file = os.path.join(dir_temp, f'batch_{batch_idx:03d}.npz')
            np.savez_compressed(batch_file, X=np.array(X_batch), y=np.array(y_batch))
    
    print(f"  ✓ Batches processados: {num_batches}")
    return dir_temp, num_batches

def verificar_e_filtrar_batches(dir_temp, num_batches, min_final=2):
    """Verifica e remove classes pequenas DOS BATCHES"""
    print(f"\n[4/6] Verificando batches...")
    
    # Conta distribuição por classe em TODOS os batches
    contagem_total = Counter()
    
    for batch_idx in tqdm(range(num_batches), desc="Analisando"):
        batch_file = os.path.join(dir_temp, f'batch_{batch_idx:03d}.npz')
        data = np.load(batch_file)
        contagem_total.update(data['y'])
    
    # Identifica classes válidas (com ≥ min_final)
    classes_validas = {classe for classe, count in contagem_total.items() 
                      if count >= min_final}
    
    print(f"  Classes totais: {len(contagem_total)}")
    print(f"  Classes válidas (≥{min_final}): {len(classes_validas)}")
    print(f"  Classes removidas: {len(contagem_total) - len(classes_validas)}")
    
    if len(classes_validas) == 0:
        raise ValueError("Nenhuma classe válida após filtro!")
    
    # Re-mapeia classes para números consecutivos
    mapeamento_novo = {classe_velha: nova for nova, classe_velha 
                      in enumerate(sorted(classes_validas))}
    
    # Processa novamente os batches filtrando e re-mapeando
    batches_filtrados = 0
    total_imagens_filtradas = 0
    
    for batch_idx in tqdm(range(num_batches), desc="Filtrando"):
        batch_file = os.path.join(dir_temp, f'batch_{batch_idx:03d}.npz')
        data = np.load(batch_file)
        X_batch = data['X']
        y_batch = data['y']
        
        # Filtra apenas classes válidas e re-mapeia
        indices_validos = [i for i, classe in enumerate(y_batch) 
                          if classe in classes_validas]
        
        if indices_validos:
            X_filtrado = X_batch[indices_validos]
            y_filtrado = y_batch[indices_validos]
            
            # Aplica novo mapeamento
            y_filtrado = np.array([mapeamento_novo[classe] for classe in y_filtrado])
            
            # Salva batch filtrado
            batch_file_filtrado = os.path.join(dir_temp, f'batch_filtrado_{batch_idx:03d}.npz')
            np.savez_compressed(batch_file_filtrado, X=X_filtrado, y=y_filtrado)
            
            batches_filtrados += 1
            total_imagens_filtradas += len(X_filtrado)
        
        # Remove batch original
        os.remove(batch_file)
    
    print(f"  Imagens após filtro: {total_imagens_filtradas:,}")
    print(f"  Classes finais: {len(classes_validas)}")
    
    return dir_temp, batches_filtrados, total_imagens_filtradas, len(classes_validas)

def dividir_batches_filtrados(dir_temp, num_batches_filtrados, total_imagens, num_classes, seed=42):
    """Divide batches já filtrados"""
    print(f"\n[5/6] Dividindo batches filtrados...")
    
    # Carrega TODOS os dados filtrados (agora cabe em memória)
    X_all = []
    y_all = []
    
    for batch_idx in tqdm(range(num_batches_filtrados), desc="Carregando"):
        batch_file = os.path.join(dir_temp, f'batch_filtrado_{batch_idx:03d}.npz')
        data = np.load(batch_file)
        X_all.append(data['X'])
        y_all.append(data['y'])
    
    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    
    print(f"  Dados carregados: {len(X):,} imagens, {num_classes} classes")
    
    # Agora faz a divisão estratificada CORRETAMENTE
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=seed
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=seed+1
    )
    
    # Limpa temporários
    import shutil
    shutil.rmtree(dir_temp)
    
    print(f"  Divisão concluída:")
    print(f"    Treino: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"    Validação: {len(X_val):,} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"    Teste: {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")
    
    # Verificação final
    print(f"  Verificação:")
    print(f"    Classes treino: {len(np.unique(y_train))}")
    print(f"    Classes val: {len(np.unique(y_val))}")
    print(f"    Classes teste: {len(np.unique(y_test))}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def salvar_resultados(X_train, y_train, X_val, y_val, X_test, y_test, mapeamento, cinza, args, num_classes):
    """Salva resultados finais"""
    print(f"\n[6/6] Salvando resultados...")
    
    dir_saida = os.path.join('..', 'Images', 'Processed')
    os.makedirs(dir_saida, exist_ok=True)
    
    # Salva arrays
    np.save(os.path.join(dir_saida, 'X_train.npy'), X_train)
    np.save(os.path.join(dir_saida, 'y_train.npy'), y_train)
    np.save(os.path.join(dir_saida, 'X_val.npy'), X_val)
    np.save(os.path.join(dir_saida, 'y_val.npy'), y_val)
    np.save(os.path.join(dir_saida, 'X_test.npy'), X_test)
    np.save(os.path.join(dir_saida, 'y_test.npy'), y_test)
    
    # Metadados
    metadados = {
        'config': {
            'percentual': args.percentual,
            'min_img': args.min_img,
            'min_final': args.min_final,
            'tamanho': args.tamanho,
            'cinza': cinza,
            'batch_size': args.batch_size,
            'seed': args.seed
        },
        'estatisticas': {
            'total_imagens': len(X_train) + len(X_val) + len(X_test),
            'classes': num_classes,
            'treino': len(X_train),
            'validacao': len(X_val),
            'teste': len(X_test),
            'percentuais': {
                'treino': f"{len(X_train)/(len(X_train)+len(X_val)+len(X_test))*100:.1f}%",
                'validacao': f"{len(X_val)/(len(X_train)+len(X_val)+len(X_test))*100:.1f}%",
                'teste': f"{len(X_test)/(len(X_train)+len(X_val)+len(X_test))*100:.1f}%"
            }
        },
        'formato_imagens': X_train.shape[1:],
        'mapeamento_classes': {str(k): v for k, v in mapeamento.items()}
    }
    
    with open(os.path.join(dir_saida, 'metadata.json'), 'w') as f:
        json.dump(metadados, f, indent=2)
    
    print(f"\n" + "="*60)
    print("PRÉ-PROCESSAMENTO CONCLUÍDO!")
    print("="*60)
    print(f"RESUMO FINAL:")
    print(f"• Imagens totais: {metadados['estatisticas']['total_imagens']:,}")
    print(f"• Classes: {metadados['estatisticas']['classes']}")
    print(f"• Formato: {tuple(metadados['formato_imagens'])}")
    print(f"• Cinza: {metadados['config']['cinza']}")
    print(f"• Treino: {metadados['estatisticas']['treino']:,} ({metadados['estatisticas']['percentuais']['treino']})")
    print(f"• Validação: {metadados['estatisticas']['validacao']:,} ({metadados['estatisticas']['percentuais']['validacao']})")
    print(f"• Teste: {metadados['estatisticas']['teste']:,} ({metadados['estatisticas']['percentuais']['teste']})")
    print("="*60)
    
    return metadados

def main():
    """Função principal corrigida"""
    global args
    args = parse_args()
    
    print("="*60)
    print(f"PRÉ-PROCESSAMENTO CORRIGIDO")
    print(f"Percentual: {args.percentual*100:.0f}% | Tamanho: {args.tamanho}")
    print(f"Cinza: {args.cinza} | Mínimo final: {args.min_final}")
    print("="*60)
    
    try:
        # 1. Carrega identidades
        pessoa_para_imagens, total = carregar_identidades()
        
        # 2. Seleciona com filtro duplo
        dataset, mapeamento = selecionar_amostra_corrigida(
            pessoa_para_imagens, total, 
            args.percentual, args.min_img, args.min_final, args.seed
        )
        
        # 3. Processa batches
        dir_temp, num_batches = processar_batches(
            dataset, args.tamanho, args.cinza, args.batch_size
        )
        
        # 4. Verifica e filtra batches (CRÍTICO!)
        dir_temp, batches_filtrados, total_imagens, num_classes = verificar_e_filtrar_batches(
            dir_temp, num_batches, args.min_final
        )
        
        # 5. Divide batches filtrados
        splits = dividir_batches_filtrados(
            dir_temp, batches_filtrados, total_imagens, num_classes, args.seed
        )
        X_train, y_train, X_val, y_val, X_test, y_test = splits
        
        # 6. Salva resultados
        metadados = salvar_resultados(
            X_train, y_train, X_val, y_val, X_test, y_test,
            mapeamento, args.cinza, args, num_classes
        )
        
    except Exception as e:
        print(f"\n❌ ERRO: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()