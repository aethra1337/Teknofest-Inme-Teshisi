"""
Colab'de Hızlı Başlangıç Scripti
Bu kodu Colab'de hücre hücre çalıştırın
"""

# ============================================================================
# ADIM 1: Proje Klasörüne Geçin
# ============================================================================
print("=" * 60)
print("ADIM 1: Proje Klasörüne Geçiliyor...")
print("=" * 60)

import os

# Proje klasörüne geçin
if os.path.exists('/content/Teknofest'):
    os.chdir('/content/Teknofest')
    print(f"✅ Proje dizini: {os.getcwd()}")
else:
    print("⚠️ Teknofest klasörü bulunamadı. Mevcut dizinler:")
    !ls -la /content/
    print("\n💡 ZIP'i açtığınızdan emin olun!")

# ============================================================================
# ADIM 2: Kütüphaneleri Kurun
# ============================================================================
print("\n" + "=" * 60)
print("ADIM 2: Kütüphaneler Kuruluyor...")
print("=" * 60)

!pip install -q tensorflow==2.16.1 keras==3.0.5
!pip install -q numpy==1.26.4 pandas==2.2.0
!pip install -q opencv-python==4.9.0.80 Pillow==10.2.0
!pip install -q scikit-learn==1.4.0 pydicom==2.4.4
!pip install -q matplotlib==3.8.2 seaborn==0.13.2

print("✅ Kütüphaneler kuruldu!")

# ============================================================================
# ADIM 3: GPU Kontrolü
# ============================================================================
print("\n" + "=" * 60)
print("ADIM 3: GPU Kontrolü")
print("=" * 60)

import tensorflow as tf

print(f"TensorFlow versiyonu: {tf.__version__}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU bulundu: {len(gpus)} adet")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i+1}: {gpu}")
    
    # GPU ayarları
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU bellek büyümesi aktif")
    except RuntimeError as e:
        print(f"⚠️ GPU ayarı hatası: {e}")
else:
    print("❌ GPU bulunamadı!")
    print("💡 Runtime → Change runtime type → GPU seçin")

# ============================================================================
# ADIM 4: Veri Setini Kontrol Edin
# ============================================================================
print("\n" + "=" * 60)
print("ADIM 4: Veri Seti Kontrolü")
print("=" * 60)

import config
from pathlib import Path

print(f"İskemi dizini: {config.ISKEMI_DIR}")
print(f"İskemi var mı: {config.ISKEMI_DIR.exists()}")

print(f"\nİnme Yok dizini: {config.INMEYOK_DIR}")
print(f"İnme Yok var mı: {config.INMEYOK_DIR.exists()}")

# PNG dosyalarını say
if config.ISKEMI_DIR.exists():
    iskemi_png = list(config.ISKEMI_DIR.rglob('*.png'))
    print(f"\n✅ İskemi PNG sayısı: {len(iskemi_png)}")
else:
    print("\n❌ İskemi dizini bulunamadı!")

if config.INMEYOK_DIR.exists():
    inmeyok_png = list(config.INMEYOK_DIR.rglob('*.png'))
    print(f"✅ İnme Yok PNG sayısı: {len(inmeyok_png)}")
else:
    print("\n❌ İnme Yok dizini bulunamadı!")

# ============================================================================
# ADIM 5: Klasörleri Oluşturun
# ============================================================================
print("\n" + "=" * 60)
print("ADIM 5: Klasörler Oluşturuluyor...")
print("=" * 60)

config.create_directories()
print("\n✅ Tüm klasörler hazır!")

# ============================================================================
# ADIM 6: Veri Setini Hazırlayın
# ============================================================================
print("\n" + "=" * 60)
print("ADIM 6: Veri Seti Hazırlanıyor...")
print("=" * 60)
print("⚠️ Bu işlem birkaç dakika sürebilir...")

from utils import data_loader

# Veri setini yükle ve böl
train_df, val_df, test_df = data_loader.load_data()

print(f"\n✅ Veri bölümleme tamamlandı:")
print(f"  Eğitim: {len(train_df)} görüntü")
print(f"  Doğrulama: {len(val_df)} görüntü")
print(f"  Test: {len(test_df)} görüntü")

# İşlenmiş veriyi hazırla
print("\n📁 Görüntüler kopyalanıyor...")
data_loader.prepare_processed_data(train_df, val_df, test_df)

print("\n✅ Veri seti hazır!")

# ============================================================================
# ADIM 7: Eğitimi Başlatın
# ============================================================================
print("\n" + "=" * 60)
print("ADIM 7: Eğitim Başlatılıyor...")
print("=" * 60)
print("⚠️ Bu işlem uzun sürebilir (20 epoch için 1-2 saat)")
print("💡 Çıktıları takip edebilirsiniz\n")

# Eğitimi başlat
!python train.py

print("\n✅ Eğitim tamamlandı!")

