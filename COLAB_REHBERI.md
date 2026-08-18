# Google Colab Kullanım Rehberi

## 🎯 Neden Google Colab?

✅ **Ücretsiz GPU** (Tesla T4, V100 gibi)  
✅ **Bilgisayarınızı yormaz**  
✅ **Daha hızlı eğitim** (GPU ile 10-20x hızlı)  
✅ **Kurulum gerektirmez**  
✅ **Kolay paylaşım**  

---

## 📋 Adım Adım Kurulum

### Adım 1: Google Colab'i Açın

1. Tarayıcınızda: https://colab.research.google.com
2. **"Yeni not defteri"** oluşturun
3. İsim verin: "TEKNOFEST_İnme_Teşhisi"

### Adım 2: GPU'yu Aktifleştirin

1. Üst menüden: **Çalışma zamanı** → **Çalışma zamanı türünü değiştir**
2. **Donanım hızlandırıcı**: **GPU** seçin
3. **Kaydet**

### Adım 3: Projeyi Yükleyin

#### Seçenek A: Google Drive'dan (Önerilen)

```python
# Hücre 1: Google Drive'ı bağlayın
from google.colab import drive
drive.mount('/content/drive')

# Proje klasörünüzü Drive'a yükleyin önce
# Sonra buraya kopyalayın:
!cp -r /content/drive/MyDrive/Teknofest /content/Teknofest
```

#### Seçenek B: GitHub'dan (Eğer repo'ya yüklediyseniz)

```python
!git clone https://github.com/kullanici-adi/Teknofest.git
```

#### Seçenek C: Manuel Yükleme

1. Colab'de: **Dosyalar** → **Klasör yükle**
2. Proje klasörünüzü seçin ve yükleyin

### Adım 4: Veri Setini Yükleyin

**Önemli:** Veri setiniz büyük olduğu için (5,557 görüntü) Google Drive'a yükleyin:

```python
# Veri setini Drive'a yükleyin (tek seferlik)
# Sonra Colab'de:

# Drive'ı bağlayın
from google.colab import drive
drive.mount('/content/drive')

# Veri setini kopyalayın (veya sembolik link oluşturun)
!ln -s /content/drive/MyDrive/Teknofest/iskemi /content/Teknofest/iskemi
!ln -s /content/drive/MyDrive/Teknofest/inmeyok /content/Teknofest/inmeyok
```

**VEYA** ZIP olarak yükleyin:

```python
# ZIP dosyasını yükleyin
from google.colab import files
uploaded = files.upload()  # ZIP dosyasını seçin

# ZIP'i açın
!unzip -q Teknofest_Veri_Seti.zip -d /content/
```

### Adım 5: Gerekli Kütüphaneleri Kurun

```python
# Hücre 2: Kütüphaneleri kurun
!pip install tensorflow==2.16.1
!pip install keras==3.0.5
!pip install numpy==1.26.4
!pip install pandas==2.2.0
!pip install opencv-python==4.9.0.80
!pip install Pillow==10.2.0
!pip install scikit-learn==1.4.0
!pip install pydicom==2.4.4
!pip install matplotlib==3.8.2
!pip install seaborn==0.13.2
```

### Adım 6: GPU Kontrolü

```python
# Hücre 3: GPU kontrolü
import tensorflow as tf
print("TensorFlow versiyonu:", tf.__version__)
print("GPU mevcut mu:", tf.config.list_physical_devices('GPU'))
```

**Beklenen çıktı:**
```
TensorFlow versiyonu: 2.16.1
GPU mevcut mu: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### Adım 7: Projeyi Çalıştırın

```python
# Hücre 4: Proje dizinine geçin
import os
os.chdir('/content/Teknofest')  # veya proje klasörünüzün yolu

# Eğitimi başlatın
!python train.py
```

---

## 🚀 Hızlı Başlangıç Template

Aşağıdaki kodu Colab'e kopyalayıp yapıştırın:

```python
# ============================================================================
# TEKNOFEST 2025 - İnme Teşhisi Modeli - Google Colab
# ============================================================================

# 1. Google Drive'ı bağlayın
from google.colab import drive
drive.mount('/content/drive')

# 2. Proje klasörüne geçin (Drive'da olmalı)
import os
os.chdir('/content/drive/MyDrive/Teknofest')  # Yolunuzu düzenleyin

# 3. Gerekli kütüphaneleri kurun
!pip install -q tensorflow==2.16.1 keras==3.0.5 numpy pandas opencv-python Pillow scikit-learn pydicom matplotlib seaborn

# 4. GPU kontrolü
import tensorflow as tf
print("GPU mevcut mu:", len(tf.config.list_physical_devices('GPU')) > 0)

# 5. Eğitimi başlatın
!python train.py
```

---

## 📊 Colab'de Çalışma Avantajları

| Özellik | Bilgisayarınız | Google Colab |
|---------|----------------|--------------|
| GPU | Kurulum gerekir | ✅ Ücretsiz |
| Hız | Yavaş (CPU) | ✅ Hızlı (GPU) |
| Bellek | Sınırlı | ✅ 12-16 GB RAM |
| Süre | Bilgisayar açık kalmalı | ✅ Tarayıcıda çalışır |
| Maliyet | Elektrik | ✅ Ücretsiz |

---

## ⚙️ Colab Ayarları

### GPU Kullanım Süresi

- **Ücretsiz:** 12 saat (kesintisiz)
- **Colab Pro:** 24 saat
- **Colab Pro+:** Öncelikli GPU

### Bellek Yönetimi

```python
# Bellek büyümesini ayarlayın
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
```

### Batch Size Ayarlama

Colab'de GPU belleği sınırlı olabilir, batch size'ı küçültün:

```python
# config.py'de veya doğrudan:
MODEL_CONFIG = {
    "batch_size": 16,  # 32 yerine 16 kullanın
    ...
}
```

---

## 💾 Sonuçları İndirme

### Modeli İndirin

```python
# Eğitim tamamlandıktan sonra
from google.colab import files

# Modeli indirin
files.download('/content/Teknofest/models/saved_models/best_model.h5')

# Sonuçları indirin
files.download('/content/Teknofest/results/metrics.csv')
files.download('/content/Teknofest/results/confusion_matrix.png')
```

### Drive'a Kaydetme

```python
# Sonuçları Drive'a kopyalayın
!cp -r /content/Teknofest/results /content/drive/MyDrive/Teknofest_Results
```

---

## 🔧 Sorun Giderme

### Problem: "GPU çalışmıyor"
**Çözüm:**
```python
# GPU'yu yeniden başlatın
# Çalışma zamanı → Çalışma zamanını yeniden başlat
```

### Problem: "Bellek yetersiz"
**Çözüm:**
- Batch size'ı küçültün (16 → 8)
- Gereksiz değişkenleri silin: `del variable_name`

### Problem: "Veri seti çok büyük"
**Çözüm:**
- ZIP olarak yükleyin
- Drive'da tutun, sembolik link kullanın

### Problem: "Oturum zaman aşımı"
**Çözüm:**
- Colab Pro kullanın (24 saat)
- Veya checkpoint'leri kaydedin, kaldığınız yerden devam edin

---

## 📝 Öneriler

1. **Checkpoint Kaydetme:**
   - Her epoch'ta model kaydediliyor
   - Oturum kapansa bile kaldığınız yerden devam edebilirsiniz

2. **TensorBoard:**
   ```python
   # TensorBoard'u Colab'de görüntüleyin
   %load_ext tensorboard
   %tensorboard --logdir results/logs
   ```

3. **İlerlemeyi İzleme:**
   ```python
   # Log dosyasını okuyun
   !tail -f results/logs/training.log
   ```

---

## 🎯 Hızlı Başlangıç Checklist

- [ ] Google Colab'i açın
- [ ] GPU'yu aktifleştirin
- [ ] Projeyi Drive'a yükleyin
- [ ] Veri setini yükleyin
- [ ] Kütüphaneleri kurun
- [ ] GPU kontrolü yapın
- [ ] `train.py` çalıştırın
- [ ] Sonuçları indirin

---

## 🆘 Yardım

- **Colab Dokümantasyon:** https://colab.research.google.com/notebooks/intro.ipynb
- **GPU Kullanımı:** https://colab.research.google.com/notebooks/gpu.ipynb
- **Drive Entegrasyonu:** https://colab.research.google.com/notebooks/snippets/drive.ipynb

---

**Not:** Colab'de çalışırken bilgisayarınızı kapatabilirsiniz! Eğitim Colab sunucularında devam eder.

