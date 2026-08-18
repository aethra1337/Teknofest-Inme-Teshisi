"""
Colab'de Tek Seferde Tüm İşlemleri Çalıştırma Scripti
Bu kodu Colab'de tek bir hücrede çalıştırın!
"""

print("=" * 70)
print("🚀 TEKNOFEST İNME TEŞHİSİ - TAM KURULUM VE EĞİTİM")
print("=" * 70)

# ============================================================================
# ADIM 1: Proje Klasörüne Geçin
# ============================================================================
print("\n" + "=" * 70)
print("ADIM 1: Proje Klasörüne Geçiliyor...")
print("=" * 70)

import os

# Proje klasörüne geçin
if os.path.exists('/content/Teknofest'):
    os.chdir('/content/Teknofest')
    print(f"✅ Proje dizini: {os.getcwd()}")
else:
    print("⚠️ Teknofest klasörü bulunamadı. Mevcut dizinler:")
    import subprocess
    subprocess.run(['ls', '-la', '/content/'])
    raise FileNotFoundError("Teknofest klasörü bulunamadı!")

# ============================================================================
# ADIM 2: Kütüphaneleri Kurun
# ============================================================================
print("\n" + "=" * 70)
print("ADIM 2: Kütüphaneler Kuruluyor...")
print("=" * 70)

import subprocess
import sys

# Önce numpy'yi uyumlu versiyona düşür (TensorFlow uyumluluğu için)
print("📚 Kütüphaneler kuruluyor...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy<1.26"], 
               check=False)

packages = [
    "tensorflow==2.16.1",
    "keras==3.0.5",
    "pandas==2.2.0",
    "opencv-python==4.9.0.80",
    "Pillow==10.2.0",
    "scikit-learn==1.4.0",
    "pydicom==2.4.4",
    "matplotlib==3.8.2",
    "seaborn==0.13.2"
]

for package in packages:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], 
                   check=False)

print("✅ Kütüphaneler kuruldu!")

# ============================================================================
# ADIM 3: GPU Kontrolü ve Ayarları
# ============================================================================
print("\n" + "=" * 70)
print("ADIM 3: GPU Kontrolü")
print("=" * 70)

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
    print("⚠️ Eğitim CPU'da çalışacak (çok yavaş olabilir)")

# ============================================================================
# ADIM 4: Veri Setini Kontrol Edin
# ============================================================================
print("\n" + "=" * 70)
print("ADIM 4: Veri Seti Kontrolü")
print("=" * 70)

import config
from pathlib import Path

print(f"İskemi dizini: {config.ISKEMI_DIR}")
print(f"İskemi var mı: {config.ISKEMI_DIR.exists()}")

print(f"\nİnme Yok dizini: {config.INMEYOK_DIR}")
print(f"İnme Yok var mı: {config.INMEYOK_DIR.exists()}")

# PNG dosyalarını say
iskemi_count = 0
inmeyok_count = 0

if config.ISKEMI_DIR.exists():
    iskemi_png = list(config.ISKEMI_DIR.rglob('*.png'))
    iskemi_count = len(iskemi_png)
    print(f"\n✅ İskemi PNG sayısı: {iskemi_count}")
else:
    print("\n❌ İskemi dizini bulunamadı!")

if config.INMEYOK_DIR.exists():
    inmeyok_png = list(config.INMEYOK_DIR.rglob('*.png'))
    inmeyok_count = len(inmeyok_png)
    print(f"✅ İnme Yok PNG sayısı: {inmeyok_count}")
else:
    print("\n❌ İnme Yok dizini bulunamadı!")

if iskemi_count == 0 or inmeyok_count == 0:
    print("\n⚠️ UYARI: Veri seti eksik görünüyor!")
    print("Eğitim devam edecek ama sonuçlar beklenmedik olabilir.")

# ============================================================================
# ADIM 5: Gerekli Klasörleri Oluşturun
# ============================================================================
print("\n" + "=" * 70)
print("ADIM 5: Klasörler Oluşturuluyor...")
print("=" * 70)

config.create_directories()
print("\n✅ Tüm klasörler hazır!")

# ============================================================================
# ADIM 6: Veri Setini Hazırlayın
# ============================================================================
print("\n" + "=" * 70)
print("ADIM 6: Veri Seti Hazırlanıyor...")
print("=" * 70)
print("⚠️ Bu işlem birkaç dakika sürebilir...")

from utils import data_loader

try:
    # Veri setini yükle ve böl
    print("📦 Veri seti yükleniyor ve bölünüyor...")
    train_df, val_df, test_df = data_loader.load_data()
    
    print(f"\n✅ Veri bölümleme tamamlandı:")
    print(f"  Eğitim: {len(train_df)} görüntü")
    print(f"  Doğrulama: {len(val_df)} görüntü")
    print(f"  Test: {len(test_df)} görüntü")
    
    # İşlenmiş veriyi hazırla
    print("\n📁 Görüntüler kopyalanıyor...")
    data_loader.prepare_processed_data(train_df, val_df, test_df)
    
    print("\n✅ Veri seti hazır!")
except Exception as e:
    print(f"\n❌ Veri hazırlama hatası: {e}")
    print("Eğitim devam edecek ama veri eksik olabilir.")
    import traceback
    traceback.print_exc()

# ============================================================================
# ADIM 7: Eğitimi Başlatın
# ============================================================================
print("\n" + "=" * 70)
print("ADIM 7: Eğitim Başlatılıyor... 🚀")
print("=" * 70)
print("⚠️ Bu işlem uzun sürebilir!")
print(f"   - Epoch sayısı: {config.MODEL_CONFIG['epochs']}")
print(f"   - Batch size: {config.MODEL_CONFIG['batch_size']}")
print(f"   - Tahmini süre: {config.MODEL_CONFIG['epochs'] * 3}-{config.MODEL_CONFIG['epochs'] * 6} dakika")
print("\n💡 Çıktıları takip edebilirsiniz...\n")

# Eğitimi başlat
try:
    import subprocess
    result = subprocess.run(
        [sys.executable, "train.py"],
        cwd=os.getcwd(),
        check=False
    )
    
    if result.returncode == 0:
        print("\n" + "=" * 70)
        print("✅ EĞİTİM TAMAMLANDI!")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("⚠️ Eğitim hata ile sonlandı (return code:", result.returncode, ")")
        print("=" * 70)
        print("Lütfen yukarıdaki hata mesajlarını kontrol edin.")
        
except Exception as e:
    print(f"\n❌ Eğitim başlatma hatası: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# ADIM 8: Sonuçları Göster
# ============================================================================
print("\n" + "=" * 70)
print("ADIM 8: Sonuçlar")
print("=" * 70)

# Eğitim özetini göster
try:
    print("\n📊 Eğitim Özeti:")
    result = subprocess.run(
        [sys.executable, "egitim_ozeti.py"],
        cwd=os.getcwd(),
        check=False
    )
except Exception as e:
    print(f"⚠️ Özet gösterilemedi: {e}")

# Test değerlendirmesi
try:
    print("\n📈 Test Değerlendirmesi:")
    result = subprocess.run(
        [sys.executable, "test_degerlendirme.py"],
        cwd=os.getcwd(),
        check=False
    )
except Exception as e:
    print(f"⚠️ Test değerlendirmesi yapılamadı: {e}")

print("\n" + "=" * 70)
print("🎉 TÜM İŞLEMLER TAMAMLANDI!")
print("=" * 70)
print("\n📁 Sonuçlar şu klasörde:")
print(f"   {os.path.join(os.getcwd(), 'results')}")
print("\n💾 Sonuçları indirmek için:")
print("   !zip -r sonuclar.zip results/")
print("   from google.colab import files")
print("   files.download('sonuclar.zip')")

