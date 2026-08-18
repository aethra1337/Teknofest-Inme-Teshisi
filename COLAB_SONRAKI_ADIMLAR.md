# 🚀 Colab'de Sonraki Adımlar

ZIP başarıyla açıldı! Şimdi şu adımları takip edin:

---

## ✅ Tamamlanan Adımlar
- [x] ZIP yüklendi
- [x] ZIP açıldı
- [x] Proje dizini ayarlandı: `/content/Teknofest`

---

## 📋 Sıradaki Adımlar

### Adım 1: Kütüphaneleri Kurun

Colab'de yeni bir hücre açın ve şu kodu çalıştırın:

```python
# Kütüphaneleri kurun
print("📚 Kütüphaneler kuruluyor...")

!pip install -q tensorflow==2.16.1 keras==3.0.5
!pip install -q numpy==1.26.4 pandas==2.2.0
!pip install -q opencv-python==4.9.0.80 Pillow==10.2.0
!pip install -q scikit-learn==1.4.0 pydicom==2.4.4
!pip install -q matplotlib==3.8.2 seaborn==0.13.2

print("✅ Kütüphaneler kuruldu!")
```

**Beklenen süre:** 2-3 dakika

---

### Adım 2: GPU Kontrolü

Yeni bir hücre açın:

```python
import tensorflow as tf

# GPU kontrolü
print("🎮 GPU Kontrolü:")
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
```

**Beklenen çıktı:** `✅ GPU bulundu: 1 adet`

---

### Adım 3: Veri Setini Kontrol Edin

Yeni bir hücre açın:

```python
import config
from pathlib import Path

print("📊 Veri Seti Kontrolü:")
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
```

**Beklenen çıktı:** Her iki dizin de `True` ve PNG sayıları görünmeli

---

### Adım 4: Gerekli Klasörleri Oluşturun

Yeni bir hücre açın:

```python
# Klasörleri oluştur
config.create_directories()
print("\n✅ Tüm klasörler hazır!")
```

---

### Adım 5: Veri Setini Hazırlayın

**⚠️ Bu adım birkaç dakika sürebilir!**

Yeni bir hücre açın:

```python
from utils import data_loader

print("📦 Veri seti hazırlanıyor...")
print("Bu işlem birkaç dakika sürebilir...")

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
print("Artık eğitime başlayabilirsiniz.")
```

**Beklenen süre:** 3-5 dakika (görüntü sayısına bağlı)

---

### Adım 6: Eğitimi Başlatın 🚀

**⚠️ Bu adım uzun sürebilir (20 epoch için 1-2 saat)!**

Yeni bir hücre açın:

```python
# Eğitimi başlat
!python train.py
```

**Beklenen süre:** 
- 3 epoch: ~15-20 dakika
- 20 epoch: ~1-2 saat

**Eğitim sırasında göreceğiniz:**
- Her epoch için loss, accuracy, F1 score
- Validation metrikleri
- Model checkpoint kayıtları

---

## 🔍 Sorun Giderme

### Problem: "ModuleNotFoundError: No module named 'config'"
**Çözüm:** Proje dizininde olduğunuzdan emin olun:
```python
import os
os.chdir('/content/Teknofest')
print(os.getcwd())
```

### Problem: "GPU bulunamadı"
**Çözüm:** 
1. Runtime → Change runtime type
2. Hardware accelerator: **GPU** seçin
3. Kaydet
4. GPU kontrolü kodunu tekrar çalıştırın

### Problem: "Veri dizini bulunamadı"
**Çözüm:** ZIP'te veri seti eksik olabilir. Kontrol edin:
```python
!ls -la /content/Teknofest/iskemi/
!ls -la /content/Teknofest/inmeyok/
```

---

## 📊 Eğitim Sonrası

Eğitim tamamlandıktan sonra:

### 1. Eğitim Özetini Görüntüleyin
```python
!python egitim_ozeti.py
```

### 2. Test Setini Değerlendirin
```python
!python test_degerlendirme.py
```

### 3. Sonuçları İndirin
```python
# Sonuçları Drive'a kaydedin
!cp -r /content/Teknofest/results /content/drive/MyDrive/Teknofest_Results
```

---

## 💡 İpuçları

1. **Hücreleri sırayla çalıştırın** - Bir hücre bitmeden diğerine geçmeyin
2. **Çıktıları takip edin** - Hata mesajlarını okuyun
3. **GPU kullanın** - Eğitim çok daha hızlı olacak
4. **Sabırlı olun** - Veri hazırlama ve eğitim zaman alır

---

**Başarılar! 🎉**

