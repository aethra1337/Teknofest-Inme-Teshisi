"""
Model Analiz Scripti
best_stroke_detection_model.h5 modelini detaylı analiz eder
"""

import os
import sys
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import numpy as np
from collections import Counter

# GPU ayarları
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[OK] {len(gpus)} GPU bulundu ve yapilandirildi")
    except RuntimeError as e:
        print(f"[UYARI] GPU yapilandirma hatasi: {e}")
else:
    print("[BILGI] GPU bulunamadi, CPU kullanilacak")

def format_bytes(size_bytes):
    """Byte'ı okunabilir formata çevirir"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def analyze_model(model_path):
    """Modeli detaylı analiz eder"""
    
    print("=" * 80)
    print("MODEL ANALİZ RAPORU")
    print("=" * 80)
    print()
    
    # Dosya bilgileri
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"[HATA] Model dosyasi bulunamadi: {model_path}")
        return
    
    file_size = model_file.stat().st_size
    print("[DOSYA] DOSYA BILGILERI")
    print("-" * 80)
    print(f"  Dosya yolu: {model_file.absolute()}")
    print(f"  Dosya boyutu: {format_bytes(file_size)}")
    print(f"  Dosya adı: {model_file.name}")
    print()
    
    # Model yükleme
    print("[YUKLEME] Model yukleniyor...")
    try:
        model = keras.models.load_model(str(model_path))
        print("[OK] Model basariyla yuklendi!")
        print()
    except Exception as e:
        print(f"[HATA] Model yukleme hatasi: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Model temel bilgileri
    print("[BILGI] MODEL TEMEL BILGILERI")
    print("-" * 80)
    print(f"  Model adı: {model.name}")
    print(f"  Toplam parametre sayısı: {model.count_params():,}")
    print(f"  Eğitilebilir parametre sayısı: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
    print(f"  Eğitilemez parametre sayısı: {sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights]):,}")
    print()
    
    # Input/Output bilgileri
    print("[IO] INPUT/OUTPUT BILGILERI")
    print("-" * 80)
    if hasattr(model, 'input_shape'):
        print(f"  Input şekli: {model.input_shape}")
    if hasattr(model, 'output_shape'):
        print(f"  Output şekli: {model.output_shape}")
    if model.inputs:
        print(f"  Input katmanı: {model.inputs[0].name}")
        print(f"  Input dtype: {model.inputs[0].dtype}")
    if model.outputs:
        print(f"  Output katmanı: {model.outputs[0].name}")
        print(f"  Output dtype: {model.outputs[0].dtype}")
    print()
    
    # Model yapısı analizi
    print("[YAPI] MODEL YAPISI ANALIZI")
    print("-" * 80)
    
    layer_types = []
    layer_info = []
    total_params = 0
    trainable_params = 0
    
    for i, layer in enumerate(model.layers):
        layer_type = type(layer).__name__
        layer_types.append(layer_type)
        
        params = layer.count_params()
        trainable = sum([tf.keras.backend.count_params(w) for w in layer.trainable_weights])
        total_params += params
        
        layer_info.append({
            'index': i,
            'name': layer.name,
            'type': layer_type,
            'params': params,
            'trainable': trainable,
            'output_shape': str(layer.output_shape) if hasattr(layer, 'output_shape') else 'N/A'
        })
    
    # Katman türü dağılımı
    layer_type_counts = Counter(layer_types)
    print("  Katman türü dağılımı:")
    for layer_type, count in sorted(layer_type_counts.items(), key=lambda x: -x[1]):
        print(f"    - {layer_type}: {count} katman")
    print()
    
    print(f"  Toplam katman sayısı: {len(model.layers)}")
    print()
    
    # Önemli katmanlar
    print("[KATMAN] ONEMLI KATMANLAR")
    print("-" * 80)
    
    # Base model bulma
    base_models = ['EfficientNet', 'ResNet', 'VGG', 'DenseNet', 'MobileNet', 'Xception', 'Inception']
    base_model_found = None
    for layer in model.layers:
        for base_name in base_models:
            if base_name in layer.name or base_name in type(layer).__name__:
                base_model_found = base_name
                break
        if base_model_found:
            break
    
    if base_model_found:
        print(f"  Base Model: {base_model_found}")
    else:
        print("  Base Model: Tespit edilemedi")
    
    # Dropout katmanları
    dropout_layers = [l for l in model.layers if 'dropout' in l.name.lower() or 'Dropout' in type(l).__name__]
    if dropout_layers:
        print(f"  Dropout katmanları: {len(dropout_layers)}")
        for dl in dropout_layers:
            if hasattr(dl, 'rate'):
                print(f"    - {dl.name}: rate={dl.rate}")
    
    # Dense/Fully Connected katmanları
    dense_layers = [l for l in model.layers if 'Dense' in type(l).__name__]
    if dense_layers:
        print(f"  Dense katmanları: {len(dense_layers)}")
        for dl in dense_layers:
            units = dl.units if hasattr(dl, 'units') else 'N/A'
            print(f"    - {dl.name}: {units} units")
    
    # Global pooling katmanları
    pooling_layers = [l for l in model.layers if 'GlobalAveragePooling' in type(l).__name__ or 'GlobalMaxPooling' in type(l).__name__]
    if pooling_layers:
        print(f"  Global Pooling katmanları: {len(pooling_layers)}")
        for pl in pooling_layers:
            print(f"    - {pl.name}: {type(pl).__name__}")
    
    print()
    
    # Model derleme bilgileri
    print("[DERLEME] MODEL DERLEME BILGILERI")
    print("-" * 80)
    if hasattr(model, 'optimizer') and model.optimizer:
        print(f"  Optimizer: {type(model.optimizer).__name__}")
        if hasattr(model.optimizer, 'learning_rate'):
            lr = model.optimizer.learning_rate
            if hasattr(lr, 'numpy'):
                print(f"  Learning Rate: {lr.numpy():.6f}")
            else:
                print(f"  Learning Rate: {lr}")
    else:
        print("  Optimizer: Model derlenmemiş")
    
    if hasattr(model, 'loss'):
        print(f"  Loss fonksiyonu: {model.loss}")
    else:
        print("  Loss fonksiyonu: Bilinmiyor")
    
    if hasattr(model, 'metrics'):
        print(f"  Metrikler: {[m.name if hasattr(m, 'name') else str(m) for m in model.metrics]}")
    print()
    
    # Model özeti (ilk 50 satır)
    print("[OZET] MODEL OZETI (Ilk 50 satir)")
    print("-" * 80)
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    
    for i, line in enumerate(summary_lines[:50]):
        print(f"  {line}")
    if len(summary_lines) > 50:
        print(f"  ... (toplam {len(summary_lines)} satır, sadece ilk 50 gösterildi)")
    print()
    
    # Katman detayları (en çok parametreye sahip 10 katman)
    print("[PARAMETRE] EN COK PARAMETREYE SAHIP 10 KATMAN")
    print("-" * 80)
    sorted_layers = sorted(layer_info, key=lambda x: x['params'], reverse=True)
    print(f"{'Sıra':<5} {'Katman Adı':<30} {'Tür':<20} {'Parametre':<15} {'Eğitilebilir':<15}")
    print("-" * 85)
    for i, layer in enumerate(sorted_layers[:10], 1):
        trainable_str = "Evet" if layer['trainable'] > 0 else "Hayır"
        print(f"{i:<5} {layer['name']:<30} {layer['type']:<20} {layer['params']:<15,} {trainable_str:<15}")
    print()
    
    # Model boyutu tahmini
    print("[BOYUT] MODEL BOYUTU TAHMINI")
    print("-" * 80)
    # Float32 için: parametre sayısı * 4 byte
    estimated_size = model.count_params() * 4
    print(f"  Tahmini model boyutu (Float32): {format_bytes(estimated_size)}")
    print(f"  Gerçek dosya boyutu: {format_bytes(file_size)}")
    compression_ratio = (1 - file_size / estimated_size) * 100 if estimated_size > 0 else 0
    print(f"  Sıkıştırma oranı: {compression_ratio:.2f}%")
    print()
    
    # Model kullanım önerileri
    print("[ONERI] MODEL KULLANIM ONERILERI")
    print("-" * 80)
    
    # Input shape kontrolü
    if model.input_shape:
        input_shape = model.input_shape[1:]  # Batch boyutunu çıkar
        print(f"  Önerilen input boyutu: {input_shape}")
    
    # Sınıf sayısı
    if model.output_shape:
        num_classes = model.output_shape[-1] if len(model.output_shape) > 1 else 1
        print(f"  Tahmin edilen sınıf sayısı: {num_classes}")
    
    # Model tipi
    if base_model_found:
        print(f"  Model tipi: Transfer Learning ({base_model_found})")
    else:
        print("  Model tipi: Özel mimari")
    
    print()
    print("=" * 80)
    print("ANALİZ TAMAMLANDI")
    print("=" * 80)

if __name__ == "__main__":
    # Model dosyası yolu
    model_path = Path("best_stroke_detection_model.h5")
    
    if not model_path.exists():
        # Alternatif konumları kontrol et
        alternative_paths = [
            Path("models/saved_models/best_stroke_detection_model.h5"),
            Path("models/best_stroke_detection_model.h5"),
            Path("saved_models/best_stroke_detection_model.h5"),
        ]
        
        found = False
        for alt_path in alternative_paths:
            if alt_path.exists():
                model_path = alt_path
                found = True
                break
        
        if not found:
            print("[HATA] Model dosyasi bulunamadi!")
            print("Lütfen model dosyasının yolunu kontrol edin.")
            sys.exit(1)
    
    analyze_model(model_path)

