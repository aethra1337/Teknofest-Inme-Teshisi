"""
🚀 GPU'ya Model Yükleme ve Test Scripti
"""

import tensorflow as tf
from pathlib import Path
import sys

print("=" * 80)
print("🚀 GPU'YA MODEL YÜKLEME VE TEST")
print("=" * 80)

# ============================================================================
# ADIM 1: GPU Kontrolü
# ============================================================================
print("\n" + "=" * 80)
print("ADIM 1: GPU Kontrolü")
print("=" * 80)

# TensorFlow versiyonu
print(f"TensorFlow Versiyonu: {tf.__version__}")

# GPU'ları listele
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU bulundu: {len(gpus)} adet")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
        try:
            details = tf.config.experimental.get_device_details(gpu)
            print(f"    Detaylar: {details}")
        except:
            pass
else:
    print("❌ GPU bulundu!")
    print("⚠️ CPU kullanılacak (çok yavaş olabilir)")

# CUDA durumu
print(f"\nCUDA Durumu:")
print(f"  CUDA Kullanılabilir: {tf.test.is_built_with_cuda()}")
print(f"  GPU Kullanılabilir: {tf.test.is_gpu_available()}")

# ============================================================================
# ADIM 2: GPU Yapılandırması
# ============================================================================
print("\n" + "=" * 80)
print("ADIM 2: GPU Yapılandırması")
print("=" * 80)

if gpus:
    try:
        # Bellek büyümesi aktif et
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU bellek büyümesi aktif")
    except RuntimeError as e:
        print(f"⚠️ GPU ayarı hatası: {e}")
        print("   (Bu normal, eğer model zaten yüklenmişse)")

# ============================================================================
# ADIM 3: GPU Test İşlemi
# ============================================================================
print("\n" + "=" * 80)
print("ADIM 3: GPU Test İşlemi")
print("=" * 80)

try:
    with tf.device('/GPU:0' if gpus else '/CPU:0'):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
        print(f"✅ Test işlemi başarılı!")
        print(f"  Sonuç: {c.numpy()}")
        print(f"  İşlem cihazı: {c.device}")
        if '/GPU' in str(c.device):
            print("  ✅ GPU'da çalışıyor!")
        else:
            print("  ⚠️ CPU'da çalışıyor")
except Exception as e:
    print(f"❌ Test hatası: {e}")

# ============================================================================
# ADIM 4: Model Yükleme (GPU'da)
# ============================================================================
print("\n" + "=" * 80)
print("ADIM 4: Model Yükleme")
print("=" * 80)

# Proje kök dizini
project_root = Path(__file__).parent
model_path = project_root / "models" / "saved_models" / "best_model.h5"

if model_path.exists():
    print(f"📁 Model dosyası: {model_path}")
    print(f"📊 Dosya boyutu: {model_path.stat().st_size / (1024*1024):.2f} MB")
    
    try:
        print("\n🔄 Model yükleniyor...")
        
        # GPU'da yükle
        with tf.device('/GPU:0' if gpus else '/CPU:0'):
            model = tf.keras.models.load_model(str(model_path))
        
        print("✅ Model başarıyla yüklendi!")
        print(f"  Model adı: {model.name}")
        print(f"  Parametre sayısı: {model.count_params():,}")
        print(f"  Yükleme cihazı: GPU" if gpus else "  Yükleme cihazı: CPU")
        
        # Model özeti
        print("\n📋 Model Özeti:")
        model.summary(print_fn=lambda x: None)  # Özeti gösterme, sadece yükleme
        
    except Exception as e:
        print(f"❌ Model yükleme hatası: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"❌ Model dosyası bulunamadı: {model_path}")
    print("⚠️ Önce modeli eğitmeniz gerekiyor: python train.py")

# ============================================================================
# ADIM 5: Tahmin Testi (GPU'da)
# ============================================================================
print("\n" + "=" * 80)
print("ADIM 5: Tahmin Testi")
print("=" * 80)

if model_path.exists() and 'model' in locals():
    try:
        # Test görüntüsü oluştur (rastgele)
        import numpy as np
        
        print("🔄 Test görüntüsü oluşturuluyor...")
        test_image = np.random.random((1, 224, 224, 3)).astype(np.float32)
        
        # GPU'da tahmin yap
        with tf.device('/GPU:0' if gpus else '/CPU:0'):
            import time
            start_time = time.time()
            predictions = model.predict(test_image, verbose=0)
            end_time = time.time()
        
        print(f"✅ Tahmin başarılı!")
        print(f"  Tahmin süresi: {(end_time - start_time)*1000:.2f} ms")
        print(f"  Tahmin şekli: {predictions.shape}")
        print(f"  Tahmin değerleri: {predictions}")
        print(f"  İşlem cihazı: {'GPU' if gpus else 'CPU'}")
        
    except Exception as e:
        print(f"❌ Tahmin hatası: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# SONUÇ
# ============================================================================
print("\n" + "=" * 80)
print("SONUÇ")
print("=" * 80)

if gpus:
    print("✅ GPU kullanılabilir ve yapılandırıldı!")
    print("✅ Model GPU'da çalışacak!")
    print("\n💡 Eğitimi başlatmak için:")
    print("   python train.py")
else:
    print("⚠️ GPU bulunamadı!")
    print("⚠️ Model CPU'da çalışacak (yavaş olabilir)")
    print("\n💡 GPU kurulumu için:")
    print("   python gpu_kontrol.py")
    print("   veya GPU_KURULUM_REHBERI.md dosyasına bakın")

print("=" * 80)

