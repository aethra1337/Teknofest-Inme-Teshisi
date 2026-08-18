"""
GPU Kontrol ve Yapılandırma Scripti
"""

import tensorflow as tf
import sys

print("=" * 80)
print("GPU KONTROL VE YAPILANDIRMA")
print("=" * 80)
print()

# TensorFlow versiyonu
print(f"TensorFlow Versiyonu: {tf.__version__}")
print()

# GPU'ları listele
print("Fiziksel GPU Cihazları:")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"  [+] {len(gpus)} GPU bulundu!")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
        try:
            # GPU detaylarını al
            gpu_details = tf.config.experimental.get_device_details(gpu)
            print(f"    Detaylar: {gpu_details}")
        except:
            pass
else:
    print("  [!] GPU bulunamadı!")
    print("  [!] CPU kullanılacak (çok yavaş olabilir)")
print()

# CUDA durumu
print("CUDA Durumu:")
print(f"  CUDA Kullanılabilir: {tf.test.is_built_with_cuda()}")
print(f"  GPU Kullanılabilir: {tf.test.is_gpu_available()}")
print()

# GPU bellek yapılandırması
if gpus:
    print("GPU Bellek Yapılandırması:")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"  [+] Memory growth aktif: {gpu}")
    except RuntimeError as e:
        print(f"  [!] Hata: {e}")
    print()

# Test: Basit bir işlem GPU'da çalışıyor mu?
print("GPU Test İşlemi:")
try:
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
        print(f"  [+] GPU'da işlem başarılı!")
        print(f"  [+] Sonuç: {c.numpy()}")
        print(f"  [+] İşlem cihazı: {c.device}")
except Exception as e:
    print(f"  [!] GPU test hatası: {e}")
    print("  [!] CPU kullanılıyor olabilir")
print()

print("=" * 80)
if gpus:
    print("SONUÇ: GPU kullanılabilir ve yapılandırıldı!")
    print("Eğitim GPU ile çalışacak (çok daha hızlı!)")
else:
    print("SONUÇ: GPU bulunamadı, CPU kullanılacak")
    print("Eğitim çok yavaş olabilir (saatler sürebilir)")
print("=" * 80)

