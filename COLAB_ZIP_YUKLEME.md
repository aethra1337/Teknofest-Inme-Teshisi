# Colab'e ZIP Dosyası Yükleme Rehberi

## 🎯 3 Farklı Yöntem

---

## Yöntem 1: Google Drive'a Yükleyip Bağlamak (ÖNERİLEN) ⭐

### Adım 1: ZIP Dosyasını Hazırlayın
```bash
# Bilgisayarınızda proje klasörünü ZIP'leyin
# Teknofest klasörünü sağ tıklayın → "Sıkıştır" veya "Send to → Compressed folder"
```

### Adım 2: Google Drive'a Yükleyin
1. https://drive.google.com adresine gidin
2. **Yeni** → **Klasör yükle** veya **Dosya yükle**
3. ZIP dosyanızı seçin ve yükleyin
4. Yükleme tamamlanana kadar bekleyin

### Adım 3: Colab'de Drive'ı Bağlayın
```python
# Colab'de bu kodu çalıştırın
from google.colab import drive
drive.mount('/content/drive')
```

### Adım 4: ZIP'i Açın
```python
import zipfile
import os

# ZIP dosyasının yolunu belirtin
zip_path = '/content/drive/MyDrive/Teknofest.zip'  # Yolunuzu düzenleyin

# ZIP'i açın
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('/content/')

# Proje klasörüne geçin
os.chdir('/content/Teknofest')
print("Mevcut dizin:", os.getcwd())
```

---

## Yöntem 2: Colab'in Dosya Yükleme Özelliği (Küçük Dosyalar İçin)

### Adım 1: Colab'de Dosya Yükleme Kodu
```python
from google.colab import files
import zipfile
import os

# ZIP dosyasını yükleyin
print("ZIP dosyasını seçin...")
uploaded = files.upload()

# Yüklenen dosyanın adını alın
zip_filename = list(uploaded.keys())[0]
print(f"Yüklenen dosya: {zip_filename}")

# ZIP'i açın
with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
    zip_ref.extractall('/content/')

# Proje klasörüne geçin
os.chdir('/content/Teknofest')
print("Mevcut dizin:", os.getcwd())
```

**⚠️ Dikkat:** Bu yöntem küçük dosyalar (<100MB) için uygundur. Büyük dosyalar için Drive yöntemini kullanın.

---

## Yöntem 3: Veri Setini Ayrı Yüklemek (Büyük Dosyalar İçin)

Eğer proje küçük ama veri seti büyükse:

### Adım 1: Projeyi ZIP'leyin (veri seti olmadan)
```bash
# Sadece kod dosyalarını ZIP'leyin
# data/ klasörünü hariç tutun
```

### Adım 2: Veri Setini Ayrı ZIP'leyin
```bash
# iskemi/ ve inmeyok/ klasörlerini ayrı ZIP'leyin
```

### Adım 3: Colab'de Her İkisini de Yükleyin
```python
# 1. Projeyi yükleyin (Drive'dan veya dosya yükleme ile)
from google.colab import drive
drive.mount('/content/drive')

# 2. Projeyi açın
import zipfile
import os

# Proje ZIP'ini aç
with zipfile.ZipFile('/content/drive/MyDrive/Teknofest_Kod.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/')

os.chdir('/content/Teknofest')

# 3. Veri setini yükleyin ve açın
from google.colab import files
print("Veri seti ZIP dosyasını seçin...")
uploaded = files.upload()

zip_filename = list(uploaded.keys())[0]
with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
    zip_ref.extractall('/content/Teknofest/')
```

---

## 🚀 Hızlı Başlangıç (Tüm Adımlar)

### Senaryo: Proje + Veri Seti Birlikte

```python
# ============================================================================
# TEK ADIMDA YÜKLEME VE KURULUM
# ============================================================================

# 1. Drive'ı bağlayın
from google.colab import drive
drive.mount('/content/drive')

# 2. ZIP'i açın
import zipfile
import os

zip_path = '/content/drive/MyDrive/Teknofest.zip'  # Yolunuzu düzenleyin

print("ZIP açılıyor...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('/content/')

# 3. Proje klasörüne geçin
os.chdir('/content/Teknofest')
print(f"✅ Proje yüklendi: {os.getcwd()}")

# 4. Dosya yapısını kontrol edin
print("\n📁 Klasör yapısı:")
!ls -la

# 5. Veri setini kontrol edin
print("\n📊 Veri seti kontrolü:")
!ls -la iskemi/iskemi/İskemi\ Veri\ Seti/PNG/ | head -5
!ls -la inmeyok/ | head -5
```

---

## 📋 Adım Adım Checklist

### Ön Hazırlık (Bilgisayarınızda):
- [ ] Proje klasörünü ZIP'leyin
- [ ] ZIP dosyasının boyutunu kontrol edin
- [ ] Google Drive'a yükleyin

### Colab'de:
- [ ] Drive'ı bağlayın
- [ ] ZIP'i açın
- [ ] Proje klasörüne geçin
- [ ] Dosya yapısını kontrol edin
- [ ] Veri setini kontrol edin

---

## 🔧 Sorun Giderme

### Problem: "ZIP dosyası çok büyük"
**Çözüm:**
- Veri setini ayrı yükleyin
- Veya Drive'da tutun, sembolik link kullanın

### Problem: "Dosya bulunamadı"
**Çözüm:**
```python
# Drive'daki dosya yolunu kontrol edin
!ls -la /content/drive/MyDrive/

# ZIP dosyasının tam yolunu bulun
import glob
zip_files = glob.glob('/content/drive/MyDrive/**/*.zip', recursive=True)
print("Bulunan ZIP dosyaları:", zip_files)
```

### Problem: "Bellek yetersiz"
**Çözüm:**
- ZIP'i açmak yerine Drive'dan doğrudan kullanın
- Sembolik link oluşturun

---

## 💡 İpuçları

1. **ZIP Boyutu:**
   - Proje kodu: ~10-50 MB
   - Veri seti: ~500 MB - 2 GB
   - Toplam: Drive'a yüklemek en iyisi

2. **Hızlı Yükleme:**
   - Drive'a yüklerken tarayıcıyı kapatmayın
   - Büyük dosyalar için Drive Desktop uygulaması kullanın

3. **Bellek Tasarrufu:**
   - ZIP'i açtıktan sonra ZIP dosyasını silin
   ```python
   !rm /content/Teknofest.zip  # Açıldıktan sonra
   ```

---

## ✅ Örnek: Tam Kurulum Scripti

```python
# ============================================================================
# TEKNOFEST PROJESİ - COLAB KURULUM
# ============================================================================

# 1. Drive bağlantısı
from google.colab import drive
drive.mount('/content/drive')

# 2. ZIP açma
import zipfile
import os

zip_path = '/content/drive/MyDrive/Teknofest.zip'
print("📦 ZIP açılıyor...")

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('/content/')

# 3. Proje dizinine geç
os.chdir('/content/Teknofest')
print(f"✅ Proje dizini: {os.getcwd()}")

# 4. Kütüphaneleri kur
print("\n📚 Kütüphaneler kuruluyor...")
!pip install -q tensorflow==2.16.1 keras==3.0.5 numpy pandas opencv-python Pillow scikit-learn pydicom matplotlib seaborn

# 5. GPU kontrolü
import tensorflow as tf
print(f"\n🎮 GPU: {len(tf.config.list_physical_devices('GPU')) > 0}")

# 6. Veri seti kontrolü
print("\n📊 Veri seti kontrol ediliyor...")
import config
print(f"İskemi: {config.ISKEMI_DIR.exists()}")
print(f"İnme Yok: {config.INMEYOK_DIR.exists()}")

print("\n✅ Kurulum tamamlandı! train.py çalıştırabilirsiniz.")
```

---

**Not:** En pratik yöntem **Google Drive'a yüklemek** ve oradan açmaktır!

