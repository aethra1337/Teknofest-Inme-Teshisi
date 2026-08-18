"""
🚀 TEKNOFEST PROJESİ - GOOGLE COLAB TAM KURULUM
Tek hücrede çalıştırılabilir - Tüm adımlar otomatik
"""

print("=" * 80)
print("🚀 TEKNOFEST PROJESİ - GOOGLE COLAB TAM KURULUM")
print("=" * 80)

# ============================================================================
# ADIM 1: Google Drive Bağlantısı
# ============================================================================
print("\n" + "=" * 80)
print("ADIM 1: Google Drive Bağlantısı")
print("=" * 80)

from google.colab import drive
drive.mount('/content/drive')
print("✅ Google Drive bağlandı")

# ============================================================================
# ADIM 2: ZIP Dosyasını Çıkart veya Klasörden Yükle
# ============================================================================
print("\n" + "=" * 80)
print("ADIM 2: Proje Yükleniyor...")
print("=" * 80)

import zipfile
import os

# ⚠️ BURAYI KENDİ YOLUNUZA GÖRE DEĞİŞTİRİN!
# Seçenek 1: ZIP dosyası varsa
ZIP_PATH = "/content/drive/MyDrive/Teknofest.zip"

# Seçenek 2: Klasör olarak yüklediyseniz
FOLDER_PATH = "/content/drive/MyDrive/Teknofest"

# Önce ZIP'i dene
if os.path.exists(ZIP_PATH):
    print("📦 ZIP dosyası bulundu, çıkartılıyor...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall("/content")
    
    extract_path = "/content/Teknofest"
    os.chdir(extract_path)
    print(f"✅ Proje ZIP'ten çıkartıldı: {os.getcwd()}")
    
elif os.path.exists(FOLDER_PATH):
    print("📁 Klasör bulundu, kullanılıyor...")
    os.chdir(FOLDER_PATH)
    print(f"✅ Proje klasörü bulundu: {os.getcwd()}")
    
else:
    print("❌ Proje bulunamadı!")
    print("\n📁 Drive'daki dosyaları kontrol ediyorum...")
    !ls -la /content/drive/MyDrive/ | head -20
    
    print("\n⚠️ LÜTFEN ŞUNLARDAN BİRİNİ YAPIN:")
    print("1. ZIP dosyasını Drive'a yükleyin: Teknofest.zip")
    print("2. Proje klasörünü Drive'a yükleyin: Teknofest/")
    print("3. Yukarıdaki ZIP_PATH veya FOLDER_PATH değişkenlerini düzenleyin")
    
    raise FileNotFoundError("Proje klasörü veya ZIP dosyası bulunamadı!")

# Dosyaları kontrol et
print("\n📁 Proje dosyaları:")
!ls -la

# ============================================================================
# ADIM 3: Kütüphaneleri Kur
# ============================================================================
print("\n" + "=" * 80)
print("ADIM 3: Kütüphaneler Kuruluyor...")
print("=" * 80)

print("📚 Kütüphaneler kuruluyor...")

# NumPy uyumsuzluğu düzelt (ÖNEMLİ!)
!pip uninstall -y numpy

# NumPy'yi uyumlu versiyona kur
!pip install -q "numpy<1.26"

# TensorFlow ve diğer kütüphaneler
!pip install -q --force-reinstall tensorflow==2.16.1
!pip install -q keras==3.0.5
!pip install -q pandas==2.2.0
!pip install -q opencv-python==4.9.0.80 Pillow==10.2.0
!pip install -q scikit-learn==1.4.0 pydicom==2.4.4
!pip install -q matplotlib==3.8.2 seaborn==0.13.2

print("✅ Kütüphaneler kuruldu!")

# Versiyon kontrolü
import numpy as np
import tensorflow as tf
print(f"\n🔍 Versiyon kontrolü:")
print(f"  NumPy: {np.__version__}")
print(f"  TensorFlow: {tf.__version__}")

# NumPy versiyon kontrolü
if int(np.__version__.split('.')[1]) >= 26:
    print("⚠️ UYARI: NumPy versiyonu çok yüksek! TensorFlow hatası olabilir.")
    print("   Runtime'ı yeniden başlatın: Runtime → Restart runtime")

# ============================================================================
# ADIM 4: GPU Kontrolü
# ============================================================================
print("\n" + "=" * 80)
print("ADIM 4: GPU Kontrolü")
print("=" * 80)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU bulundu: {len(gpus)} adet")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i+1}: {gpu}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU bellek büyümesi aktif")
    except RuntimeError as e:
        print(f"⚠️ GPU ayarı hatası: {e}")
else:
    print("❌ GPU bulunamadı!")
    print("⚠️ Runtime → Change runtime type → GPU seçin")
    print("⚠️ CPU ile çalışacak, çok yavaş olabilir!")

# ============================================================================
# ADIM 5: Proje Modüllerini Test Et
# ============================================================================
print("\n" + "=" * 80)
print("ADIM 5: Proje Modülleri Test Ediliyor...")
print("=" * 80)

import sys
current_dir = os.getcwd()
sys.path.insert(0, current_dir)

try:
    import config
    print("✅ config.py yüklendi")
    
    # Veri dizinlerini kontrol et
    print(f"\n📊 Veri dizinleri:")
    print(f"  İskemi: {config.ISKEMI_DIR}")
    print(f"  İskemi var mı: {config.ISKEMI_DIR.exists()}")
    print(f"  İnme Yok: {config.INMEYOK_DIR}")
    print(f"  İnme Yok var mı: {config.INMEYOK_DIR.exists()}")
    
    from utils import data_loader
    print("✅ utils.data_loader yüklendi")
    
    from models import model_builder
    print("✅ models.model_builder yüklendi")
    
    from training import trainer
    print("✅ training.trainer yüklendi")
    
    from evaluation import evaluator
    print("✅ evaluation.evaluator yüklendi")
    
    print("\n" + "=" * 80)
    print("🎉 TÜM KURULUM TAMAMLANDI!")
    print("=" * 80)
    print(f"📁 Çalışma dizini: {os.getcwd()}")
    print("\n✅ Artık eğitimi başlatabilirsiniz:")
    print("\n   Seçenek 1: Normal eğitim")
    print("   !python train.py")
    print("\n   Seçenek 2: Akut iyileştirme eğitimi (25 epoch)")
    print("   !python colab_train_akut_iyilestirme.py")
    print("\n" + "=" * 80)
    
except ImportError as e:
    print(f"❌ Hata: {e}")
    print("\n⚠️ Proje dosyalarını kontrol edin!")
    print("📁 Mevcut dizin:", os.getcwd())
    print("\n📂 Dosya listesi:")
    !ls -la
    
    print("\n🔍 Eksik dosyalar:")
    required_files = [
        "config.py",
        "train.py",
        "utils/data_loader.py",
        "models/model_builder.py",
        "training/trainer.py",
        "evaluation/evaluator.py"
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - BULUNAMADI!")

print("\n✅ Kurulum scripti tamamlandı!")


