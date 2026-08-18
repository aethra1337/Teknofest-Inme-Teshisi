# 📤 Google Colab'e Proje Yükleme Rehberi

## 🎯 Yöntem 1: ZIP Dosyası ile (EN KOLAY) ⭐

### Adım 1: Projeyi ZIP'le

**Windows'ta:**
1. Proje klasörünü sağ tıklayın: `C:\Users\LENOVO\Desktop\Projeler\Teknofest`
2. **"Sıkıştır"** veya **"Send to → Compressed (zipped) folder"** seçin
3. `Teknofest.zip` dosyası oluşacak

**Alternatif (PowerShell):**
```powershell
cd "C:\Users\LENOVO\Desktop\Projeler"
Compress-Archive -Path "Teknofest" -DestinationPath "Teknofest.zip"
```

### Adım 2: Google Drive'a Yükle

1. [Google Drive](https://drive.google.com) açın
2. **"Yeni"** → **"Dosya yükle"** tıklayın
3. `Teknofest.zip` dosyasını seçin
4. Yükleme tamamlanana kadar bekleyin

### Adım 3: Colab'de Aç

**Yeni bir Colab notebook oluşturun ve şu kodu çalıştırın:**

```python
# ============================================================================
# ADIM 1: Google Drive Bağlantısı
# ============================================================================
from google.colab import drive
drive.mount('/content/drive')
print("✅ Google Drive bağlandı")

# ============================================================================
# ADIM 2: ZIP Dosyasını Çıkart
# ============================================================================
import zipfile
import os

# ZIP dosyasının yolunu belirtin (Drive'daki konumunuza göre değiştirin)
zip_path = "/content/drive/MyDrive/Teknofest.zip"  # ⚠️ KENDİ YOLUNUZA GÖRE DEĞİŞTİRİN!

# Çıkartma dizini
extract_path = "/content/Teknofest"

# ZIP'i çıkart
if os.path.exists(zip_path):
    print("📦 ZIP dosyası çıkartılıyor...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("/content")
    print(f"✅ Proje çıkartıldı: {extract_path}")
    
    # Proje dizinine geç
    os.chdir(extract_path)
    print(f"✅ Çalışma dizini: {os.getcwd()}")
    
    # Dosyaları kontrol et
    print("\n📁 Proje dosyaları:")
    !ls -la
else:
    print(f"❌ ZIP dosyası bulunamadı: {zip_path}")
    print("📁 Drive'daki dosyaları kontrol ediyorum...")
    !ls -la /content/drive/MyDrive/*.zip
```

### Adım 4: Kütüphaneleri Kur

```python
# ============================================================================
# ADIM 3: Kütüphaneleri Kur
# ============================================================================
print("📚 Kütüphaneler kuruluyor...")

# NumPy uyumsuzluğu düzelt
!pip uninstall -y numpy
!pip install -q "numpy<1.26"

# TensorFlow ve diğer kütüphaneler
!pip install -q --force-reinstall tensorflow==2.16.1
!pip install -q keras==3.0.5
!pip install -q pandas==2.2.0
!pip install -q opencv-python==4.9.0.80 Pillow==10.2.0
!pip install -q scikit-learn==1.4.0 pydicom==2.4.4
!pip install -q matplotlib==3.8.2 seaborn==0.13.2

print("✅ Kütüphaneler kuruldu!")
```

### Adım 5: Eğitimi Başlat

```python
# ============================================================================
# ADIM 4: Eğitimi Başlat
# ============================================================================
# colab_train_akut_iyilestirme.py dosyasını çalıştırın
# veya direkt train.py'yi çalıştırın

!python train.py
```

---

## 🎯 Yöntem 2: Google Drive'a Klasör Olarak Yükleme

### Adım 1: Google Drive Desktop Uygulaması

1. [Google Drive Desktop](https://www.google.com/drive/download/) indirin ve kurun
2. Proje klasörünüzü Drive'a sürükleyin
3. Senkronizasyon tamamlanana kadar bekleyin

### Adım 2: Colab'de Erişim

```python
from google.colab import drive
drive.mount('/content/drive')

# Proje yolunu belirtin
PROJECT_PATH = "/content/drive/MyDrive/Teknofest"  # ⚠️ KENDİ YOLUNUZA GÖRE DEĞİŞTİRİN!

import os
os.chdir(PROJECT_PATH)
print(f"✅ Proje dizini: {os.getcwd()}")

# Dosyaları kontrol et
!ls -la
```

---

## 🎯 Yöntem 3: GitHub ile (Geliştiriciler İçin)

### Adım 1: GitHub'a Yükle

```bash
# Git ile (eğer git kuruluysa)
cd C:\Users\LENOVO\Desktop\Projeler\Teknofest
git init
git add .
git commit -m "Initial commit"
# GitHub'da yeni repo oluşturup:
git remote add origin https://github.com/kullanici_adi/Teknofest.git
git push -u origin main
```

### Adım 2: Colab'de Klonla

```python
# GitHub'dan klonla
!git clone https://github.com/kullanici_adi/Teknofest.git
os.chdir("/content/Teknofest")
```

---

## 🚀 TEK SEFERDE TÜM KOD (Önerilen)

Aşağıdaki kodu Colab'de **tek bir hücrede** çalıştırın:

```python
"""
🚀 TEKNOFEST PROJESİ - TAM KURULUM
Google Colab için tek seferde kurulum
"""

print("=" * 80)
print("🚀 TEKNOFEST PROJESİ - GOOGLE COLAB KURULUMU")
print("=" * 80)

# ============================================================================
# ADIM 1: Google Drive Bağlantısı
# ============================================================================
from google.colab import drive
drive.mount('/content/drive')
print("✅ Google Drive bağlandı")

# ============================================================================
# ADIM 2: ZIP Dosyasını Çıkart
# ============================================================================
import zipfile
import os

# ⚠️ ZIP DOSYASININ YOLUNU KENDİ YOLUNUZA GÖRE DEĞİŞTİRİN!
zip_path = "/content/drive/MyDrive/Teknofest.zip"

if os.path.exists(zip_path):
    print("\n📦 ZIP dosyası çıkartılıyor...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("/content")
    
    extract_path = "/content/Teknofest"
    os.chdir(extract_path)
    print(f"✅ Proje çıkartıldı: {os.getcwd()}")
else:
    # Alternatif: Drive'dan direkt klasör
    extract_path = "/content/drive/MyDrive/Teknofest"  # ⚠️ KENDİ YOLUNUZA GÖRE DEĞİŞTİRİN!
    if os.path.exists(extract_path):
        os.chdir(extract_path)
        print(f"✅ Proje klasörü bulundu: {os.getcwd()}")
    else:
        print(f"❌ Proje bulunamadı!")
        print("📁 Drive'daki dosyaları kontrol ediyorum...")
        !ls -la /content/drive/MyDrive/
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

# NumPy uyumsuzluğu düzelt
!pip uninstall -y numpy
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

# ============================================================================
# ADIM 4: GPU Kontrolü
# ============================================================================
print("\n" + "=" * 80)
print("ADIM 4: GPU Kontrolü")
print("=" * 80)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU bulundu: {len(gpus)} adet")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("❌ GPU bulunamadı! Runtime → Change runtime type → GPU seçin")

# ============================================================================
# ADIM 5: Proje Modüllerini Test Et
# ============================================================================
print("\n" + "=" * 80)
print("ADIM 5: Proje Modülleri Test Ediliyor...")
print("=" * 80)

import sys
sys.path.insert(0, os.getcwd())

try:
    import config
    print("✅ config.py yüklendi")
    
    from utils import data_loader
    print("✅ utils.data_loader yüklendi")
    
    from models import model_builder
    print("✅ models.model_builder yüklendi")
    
    print("\n🎉 TÜM KURULUM TAMAMLANDI!")
    print(f"📁 Çalışma dizini: {os.getcwd()}")
    print("\n✅ Artık eğitimi başlatabilirsiniz:")
    print("   !python train.py")
    print("   veya")
    print("   !python colab_train_akut_iyilestirme.py")
    
except ImportError as e:
    print(f"❌ Hata: {e}")
    print("⚠️ Proje dosyalarını kontrol edin!")

print("\n" + "=" * 80)
```

---

## 📋 Adım Adım Özet

### En Kolay Yöntem (ZIP):

1. ✅ Projeyi ZIP'le (`Teknofest.zip`)
2. ✅ Google Drive'a yükle
3. ✅ Colab'de yukarıdaki kodu çalıştır
4. ✅ ZIP yolunu düzenle (`/content/drive/MyDrive/Teknofest.zip`)
5. ✅ Shift+Enter ile çalıştır

### ZIP Dosyası Oluşturma (PowerShell):

```powershell
# PowerShell'de çalıştırın
cd "C:\Users\LENOVO\Desktop\Projeler"
Compress-Archive -Path "Teknofest" -DestinationPath "Teknofest.zip" -Force
```

---

## ⚠️ Önemli Notlar

1. **ZIP Yolu**: Drive'daki ZIP dosyasının tam yolunu yazın
2. **Klasör Yolu**: Eğer klasör olarak yüklediyseniz, klasör yolunu yazın
3. **GPU**: Runtime → Change runtime type → GPU seçin
4. **Runtime Restart**: Kütüphane kurulumundan sonra runtime'ı yeniden başlatın

---

## 🐛 Sorun Giderme

### ZIP Bulunamadı
```python
# Drive'daki tüm ZIP dosyalarını listele
!ls -la /content/drive/MyDrive/*.zip
```

### Klasör Bulunamadı
```python
# Drive'daki klasörleri listele
!ls -la /content/drive/MyDrive/
```

### Dosya Yolu Hatası
- Drive'daki tam yolu kontrol edin
- Büyük/küçük harf duyarlılığına dikkat edin
- Türkçe karakter sorunları olabilir

---

## ✅ Başarı Kontrolü

Kurulum başarılıysa şunları görmelisiniz:

```
✅ Google Drive bağlandı
✅ Proje çıkartıldı: /content/Teknofest
✅ Kütüphaneler kuruldu!
✅ config.py yüklendi
✅ utils.data_loader yüklendi
✅ models.model_builder yüklendi
🎉 TÜM KURULUM TAMAMLANDI!
```

Artık eğitimi başlatabilirsiniz! 🚀


