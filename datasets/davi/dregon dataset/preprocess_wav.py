import os
import librosa
import soundfile as sf
import numpy as np

# Configurações padrão para WaveGAN
TARGET_SR = 16000
WINDOW_SIZE = 16384  # 2^14
OVERLAP = 0.5        # 50% de sobreposição
OUTPUT_FOLDER_NAME = 'processed'

def process_audio_files_recursively(base_dir='.'):
    """
    Busca arquivos .wav recursivamente. Para cada pasta com .wavs,
    cria um subdiretório 'processed' e salva os recortes nele.
    """
    
    # os.walk percorre a pasta base e todas as pastas filhas
    for root, dirs, files in os.walk(base_dir):
        
        # TRAVA DE SEGURANÇA: Impede que o script entre nas pastas 'processed'
        # que ele mesmo acabou de criar, evitando processamento duplicado.
        if OUTPUT_FOLDER_NAME in dirs:
            dirs.remove(OUTPUT_FOLDER_NAME)

        # Filtra apenas os arquivos .wav na pasta atual
        wav_files = [f for f in files if f.lower().endswith('.wav')]
        
        # Se não tiver .wav nessa pasta, pula para a próxima
        if not wav_files:
            continue
            
        print(f"\n[{root}] -> Encontrados {len(wav_files)} arquivos. Processando...")

        # Cria a pasta 'processed' DENTRO do diretório atual (root)
        output_dir = os.path.join(root, OUTPUT_FOLDER_NAME)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Processa os arquivos encontrados nesta pasta
        for file_name in wav_files:
            file_path = os.path.join(root, file_name)
            
            try:
                audio, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
                audio = librosa.util.normalize(audio)

                step = int(WINDOW_SIZE * (1 - OVERLAP))
                num_segments = 0
                
                # Se o áudio original for menor que 1 segundo, preenche com silêncio
                if len(audio) < WINDOW_SIZE:
                    audio = np.pad(audio, (0, WINDOW_SIZE - len(audio)), 'constant')
                
                for i in range(0, len(audio) - WINDOW_SIZE + 1, step):
                    segment = audio[i : i + WINDOW_SIZE]
                    
                    output_name = f"{os.path.splitext(file_name)[0]}_seg{num_segments}.wav"
                    sf.write(os.path.join(output_dir, output_name), segment, TARGET_SR)
                    num_segments += 1
                
                print(f"  ↳ {file_name}: {num_segments} segmentos gerados.")

            except Exception as e:
                print(f"  [ERRO] Falha ao processar {file_name}: {e}")

if __name__ == '__main__':
    # Roda a partir do diretório onde o script está localizado
    process_audio_files_recursively()