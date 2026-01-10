#!/usr/bin/env python3
"""
CONDITIONAL GAN (cGAN) PARA GERAÇÃO DE FACES - CORRIGIDO
"""

import os
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import argparse
import sys

class ConditionalGAN:
    """cGAN para geração de faces condicionadas por identidade"""
    
    def __init__(self, latent_dim=100, img_shape=(64, 64, 1), num_classes=1687):
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.num_classes = num_classes
        
        # Construir gerador e discriminador
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
    def build_generator(self):
        """Constrói o gerador - CORRIGIDO para saída 64x64"""
        noise = keras.Input(shape=(self.latent_dim,))
        label = keras.Input(shape=(1,), dtype='int32')
        
        # Embedding do label
        label_embedding = keras.layers.Embedding(self.num_classes, self.latent_dim)(label)
        label_embedding = keras.layers.Flatten()(label_embedding)
        
        # Concatenar ruído + label
        model_input = keras.layers.Concatenate()([noise, label_embedding])
        
        # Camadas do gerador - CORRIGIDO para saída 64x64
        # Começa com 4x4 e faz upsample para 64x64
        x = keras.layers.Dense(4 * 4 * 512, use_bias=False)(model_input)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Reshape((4, 4, 512))(x)
        
        # Upsample 1: 4x4 -> 8x8
        x = keras.layers.Conv2DTranspose(256, 4, strides=2, padding='same', use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        
        # Upsample 2: 8x8 -> 16x16
        x = keras.layers.Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        
        # Upsample 3: 16x16 -> 32x32
        x = keras.layers.Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        
        # Upsample 4: 32x32 -> 64x64
        x = keras.layers.Conv2DTranspose(32, 4, strides=2, padding='same', use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        
        # Saída final (64x64) - CORRIGIDO: remover um Conv2DTranspose extra
        img = keras.layers.Conv2DTranspose(self.img_shape[2], 4, padding='same', 
                                          activation='tanh')(x)  # NOTA: strides=1
        
        return keras.Model([noise, label], img, name='generator')
    
    def build_discriminator(self):
        """Constrói o discriminador"""
        img = keras.Input(shape=self.img_shape)
        label = keras.Input(shape=(1,), dtype='int32')
        
        # Embedding do label
        label_embedding = keras.layers.Embedding(
            self.num_classes, 
            self.img_shape[0] * self.img_shape[1]
        )(label)
        label_embedding = keras.layers.Reshape((self.img_shape[0], self.img_shape[1], 1))(label_embedding)
        
        # Concatenar imagem + label
        x = keras.layers.Concatenate(axis=-1)([img, label_embedding])
        
        # Camadas do discriminador - SIMPLIFICADO para economia de memória
        x = keras.layers.Conv2D(64, 4, strides=2, padding='same')(x)
        x = keras.layers.LeakyReLU(0.2)(x)
        x = keras.layers.Dropout(0.3)(x)
        
        x = keras.layers.Conv2D(128, 4, strides=2, padding='same')(x)
        x = keras.layers.LeakyReLU(0.2)(x)
        x = keras.layers.Dropout(0.3)(x)
        
        x = keras.layers.Conv2D(256, 4, strides=2, padding='same')(x)
        x = keras.layers.LeakyReLU(0.2)(x)
        x = keras.layers.Dropout(0.3)(x)
        
        x = keras.layers.Flatten()(x)
        validity = keras.layers.Dense(1, activation='sigmoid')(x)
        
        return keras.Model([img, label], validity, name='discriminator')
    
    def train(self, X_train, y_train, epochs=50, batch_size=32, sample_interval=10):
        """Treina a cGAN"""
        
        # Compilar discriminador
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            metrics=['accuracy']
        )
        
        # Congelar discriminador durante treino do gerador
        self.discriminator.trainable = False
        
        # cGAN combinada (gerador + discriminador congelado)
        noise = keras.Input(shape=(self.latent_dim,))
        label = keras.Input(shape=(1,))
        img = self.generator([noise, label])
        validity = self.discriminator([img, label])
        self.combined = keras.Model([noise, label], validity)
        self.combined.compile(
            loss='binary_crossentropy',
            optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        )
        
        # Labels
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        # Histórico de perdas
        d_losses = []
        g_losses = []
        d_accs = []
        
        print(f"\nIniciando treinamento por {epochs} épocas...")
        print(f"Batch size: {batch_size}")
        print(f"Classes: {self.num_classes}")
        print("-" * 50)
        
        for epoch in range(epochs):
            # ---------------------
            #  Treinar Discriminador
            # ---------------------
            # Selecionar batch aleatório de imagens reais
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_imgs = X_train[idx]
            labels = y_train[idx].reshape(-1, 1)
            
            # Gerar batch de imagens falsas
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_imgs = self.generator.predict([noise, labels], verbose=0)
            
            # Treinar discriminador
            d_loss_real = self.discriminator.train_on_batch([real_imgs, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([fake_imgs, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # ---------------------
            #  Treinar Gerador
            # ---------------------
            # Gerar ruído e labels aleatórios
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            sampled_labels = np.random.randint(0, self.num_classes, batch_size).reshape(-1, 1)
            
            # Treinar gerador (quer enganar discriminador)
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)
            
            # Armazenar métricas
            d_losses.append(d_loss[0])
            g_losses.append(g_loss)
            d_accs.append(d_loss[1])
            
            # Mostrar progresso
            if epoch % 1 == 0:  # Mostrar toda época para monitoramento
                print(f"Época {epoch:4d}/{epochs} | "
                      f"D-loss: {d_loss[0]:.4f} | "
                      f"D-acc: {d_loss[1]:.2%} | "
                      f"G-loss: {g_loss:.4f}")
            
            # Salvar amostras periodicamente
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
        
        return d_losses, g_losses, d_accs
    
    def sample_images(self, epoch, grid_size=4):
        """Gera e salva amostras de imagens"""
        # Configurar plot
        rows = cols = grid_size
        fig, axs = plt.subplots(rows, cols, figsize=(8, 8))
        
        # Gerar imagens
        noise = np.random.normal(0, 1, (rows * cols, self.latent_dim))
        labels = np.random.randint(0, min(100, self.num_classes), rows * cols).reshape(-1, 1)
        gen_imgs = self.generator.predict([noise, labels], verbose=0)
        
        # Converter de [-1, 1] para [0, 1] para visualização
        gen_imgs = 0.5 * gen_imgs + 0.5
        
        # Plotar
        cnt = 0
        for i in range(rows):
            for j in range(cols):
                if self.img_shape[2] == 1:  # Grayscale
                    axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                else:  # RGB
                    axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                
                axs[i, j].set_title(f"Classe {labels[cnt][0]}")
                axs[i, j].axis('off')
                cnt += 1
        
        # Salvar figura
        os.makedirs('../Resultados/cgan_samples', exist_ok=True)
        fig.savefig(f'../Resultados/cgan_samples/epoch_{epoch:04d}.png')
        plt.close()
        
        print(f"  Amostras salvas: ../Resultados/cgan_samples/epoch_{epoch:04d}.png")
    
    def generate_for_class(self, class_id, num_images=10):
        """Gera imagens para uma classe específica"""
        noise = np.random.normal(0, 1, (num_images, self.latent_dim))
        labels = np.full((num_images, 1), class_id, dtype=int)
        gen_imgs = self.generator.predict([noise, labels], verbose=0)
        
        # Converter de [-1, 1] para [0, 1]
        gen_imgs = 0.5 * gen_imgs + 0.5
        
        return gen_imgs

def load_data():
    """Carrega dados pré-processados"""
    print("Carregando dados...")
    
    try:
        X_train = np.load('../Images/Processed/X_train.npy')
        y_train = np.load('../Images/Processed/y_train.npy')
        
        with open('../Images/Processed/metadata.json', 'r') as f:
            metadata = json.load(f)
        
        print(f"✓ Dados carregados:")
        print(f"  Imagens: {X_train.shape[0]:,}")
        print(f"  Classes: {metadata['estatisticas']['classes']}")
        print(f"  Formato: {X_train.shape[1:]}")
        
        return X_train, y_train, metadata
        
    except Exception as e:
        print(f"❌ Erro ao carregar dados: {e}")
        sys.exit(1)

def plot_training_history(d_losses, g_losses, d_accs):
    """Plota histórico de treinamento"""
    if len(d_losses) == 0:
        print("⚠  Nenhum dado de treinamento para plotar")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    ax1.plot(d_losses, label='Discriminador')
    ax1.plot(g_losses, label='Gerador')
    ax1.set_title('Perdas durante treinamento')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(d_accs)
    ax2.set_title('Acurácia do Discriminador')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Acurácia')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('../Resultados/cgan_training_history.png', dpi=150)
    plt.close()
    print(f"✓ Histórico salvo: ../Resultados/cgan_training_history.png")

def main():
    """Função principal para treinar a cGAN"""
    parser = argparse.ArgumentParser(description='Treinar cGAN para geração de faces')
    parser.add_argument('--epochs', type=int, default=5, help='Número de épocas')
    parser.add_argument('--batch_size', type=int, default=8, help='Tamanho do batch')
    parser.add_argument('--latent_dim', type=int, default=100, help='Dimensão latente')
    
    args = parser.parse_args()
    
    print("="*70)
    print("TREINAMENTO cGAN - GERAÇÃO DE FACES SINTÉTICAS")
    print("="*70)
    
    # 1. Carregar dados
    X_train, y_train, metadata = load_data()
    
    # Converter dados para formato da cGAN
    X_train_gan = X_train * 2.0 - 1.0  # [0,1] -> [-1,1]
    y_train_gan = y_train.reshape(-1, 1)
    
    # 2. Criar cGAN
    print(f"\nCriando cGAN...")
    print(f"  Dimensão latente: {args.latent_dim}")
    print(f"  Formato imagem: {X_train.shape[1:]}")
    print(f"  Número de classes: {metadata['estatisticas']['classes']}")
    
    cgan = ConditionalGAN(
        latent_dim=args.latent_dim,
        img_shape=X_train.shape[1:],
        num_classes=metadata['estatisticas']['classes']
    )
    
    # Mostrar resumos
    print("\nResumo do Gerador:")
    print(f"  Total params: {cgan.generator.count_params():,}")
    
    print("\nResumo do Discriminador:")
    print(f"  Total params: {cgan.discriminator.count_params():,}")
    
    # 3. Verificar formato da saída do gerador
    print("\nVerificando formato do gerador...")
    test_noise = np.random.normal(0, 1, (1, args.latent_dim))
    test_label = np.array([[0]])
    test_output = cgan.generator.predict([test_noise, test_label], verbose=0)
    print(f"  Formato da saída do gerador: {test_output.shape}")
    print(f"  Esperado: (1, 64, 64, 1)")
    
    if test_output.shape[1:] != X_train.shape[1:]:
        print(f"❌ ERRO: Gerador produz imagens {test_output.shape[1:]}, esperado {X_train.shape[1:]}")
        print("  Ajustando arquitetura...")
        return
    
    # 4. Treinar cGAN
    print(f"\nIniciando treinamento ({args.epochs} épocas)...")
    print("Atenção: Treinamento de GANs é instável e pode demorar.")
    
    try:
        d_losses, g_losses, d_accs = cgan.train(
            X_train=X_train_gan,
            y_train=y_train_gan,
            epochs=args.epochs,
            batch_size=args.batch_size,
            sample_interval=2  # Salvar mais frequentemente para testes
        )
        
        print("\n✓ Treinamento concluído com sucesso!")
        
    except Exception as e:
        print(f"\n❌ Erro durante treinamento: {e}")
        print("Possíveis soluções:")
        print("1. Reduzir batch_size para 4 ou 2")
        print("2. Reduzir número de épocas")
        print("3. Verificar memória disponível")
        return
    
    # 5. Salvar modelos
    print("\nSalvando modelos...")
    os.makedirs('../Resultados', exist_ok=True)
    
    try:
        cgan.generator.save('../Resultados/cgan_generator.h5')
        cgan.discriminator.save('../Resultados/cgan_discriminator.h5')
        
        print("✓ Modelos salvos em ../Resultados/")
        print("  - cgan_generator.h5")
        print("  - cgan_discriminator.h5")
        
    except Exception as e:
        print(f"⚠  Erro ao salvar modelos: {e}")
    
    # 6. Plotar histórico
    plot_training_history(d_losses, g_losses, d_accs)
    
    # 7. Gerar amostras finais
    print("\nGerando amostras finais...")
    try:
        cgan.sample_images(epoch='final')
    except:
        print("⚠  Não foi possível gerar amostras finais")
    
    print("\n" + "="*70)
    print("TREINAMENTO cGAN CONCLUÍDO!")
    print("="*70)
    if d_losses:
        print(f"Épocas treinadas: {len(d_losses)}")
        print(f"Perda final - Discriminador: {d_losses[-1]:.4f}")
        print(f"Perda final - Gerador: {g_losses[-1]:.4f}")
        print(f"Acurácia final do Discriminador: {d_accs[-1]:.2%}")
    print("\nArquivos gerados:")
    print("  • ../Resultados/cgan_generator.h5 (se salvou)")
    print("  • ../Resultados/cgan_training_history.png (se treinou)")
    print("  • ../Resultados/cgan_samples/ (amostras)")
    print("="*70)

if __name__ == "__main__":
    # Configurar para evitar warnings
    import warnings
    warnings.filterwarnings('ignore')
    
    # Executar
    main()