# 🚀 GPU'ya Model Yükleme Rehberi

## 📋 İçindekiler

1. [Yerel Bilgisayarda GPU Kullanımı](#yerel-bilgisayarda-gpu-kullanımı)
2. [Google Colab'de GPU Kullanımı](#google-colabde-gpu-kullanımı)
3. [Modeli GPU'da Çalıştırma](#modeli-gpuda-çalıştırma)
4. [GPU Kontrolü](#gpu-kontrolü)

---

## 🖥️ Yerel Bilgisayarda GPU Kullanımı

### Adım 1: GPU Kontrolü

```python
# gpu_kontrol.py dosyasını çalıştırın
python gpu_kontrol.py
```

**Veya manuel kontrol:**

```python
import tensorflow as tf

# GPU'ları listele
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU sayısı: {len(gpus)}")

if gpus:
    for i, gpu in enumerate(gpus):
        print(f"GPU {i}: {gpu}")
else:
    print("❌ GPU bulunamadı!")
```

### Adım 2: GPU Yapılandırması

`config.py` dosyasında GPU ayarlarını kontrol edin:

```python
GPU_CONFIG = {
    "use_gpu": True,  # ✅ True olmalı
    "gpu_memory_growth": True,  # ✅ True olmalı
    "mixed_precision": False,
}
```

### Adım 3: GPU'da Eğitim Başlatma

```python
# train.py otomatik olarak GPU kullanır
python train.py
```

**Eğitim sırasında GPU kullanımını kontrol edin:**

```powershell
# Başka bir terminalde
nvidia-smi
```

---

## ☁️ Google Colab'de GPU Kullanımı

### Adım 1: GPU'yu Aktifleştir

1. Colab notebook'u açın
2. **Runtime → Change runtime type**
3. **Hardware accelerator:** **GPU** seçin
4. **Save** tıklayın

### Adım 2: GPU Kontrolü

```python
import tensorflow as tf

# GPU kontrolü
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU bulundu: {len(gpus)}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("✅ GPU yapılandırıldı")
else:
    print("❌ GPU bulunamadı! Runtime → Change runtime type → GPU")
```

### Adım 3: Eğitimi Başlat

```python
# Eğitimi başlat
!python train.py
```

---

## 🎯 Modeli GPU'da Çalıştırma

### Yöntem 1: Otomatik (Önerilen)

Model otomatik olarak GPU'da çalışır (GPU varsa):

```python
import tensorflow as tf
from models.stroke_classifier import StrokeClassifier
from pathlib import Path

# GPU yapılandırması
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Modeli yükle
model = StrokeClassifier()
model.load_model(Path("models/saved_models/best_model.h5"))

# Model otomatik olarak GPU'da çalışacak
predictions = model.predict(images)
```

### Yöntem 2: Manuel GPU Seçimi

```python
import tensorflow as tf

# GPU'yu açıkça belirt
with tf.device('/GPU:0'):
    # Modeli yükle
    model = tf.keras.models.load_model("models/saved_models/best_model.h5")
    
    # Tahmin yap
    predictions = model.predict(images)
```

### Yöntem 3: Model Oluşturma (GPU'da)

```python
import tensorflow as tf
from models.model_builder import build_model

# GPU yapılandırması
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Model oluştur (GPU'da otomatik çalışır)
model = build_model(
    input_shape=(224, 224, 3),
    num_classes=2,
    base_model_name="EfficientNetB3",
    dropout_rate=0.5,
    weights="imagenet"
)

# Modeli derle
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Model GPU'da çalışacak
```

---

## 🔍 GPU Kontrolü ve Test

### Hızlı GPU Testi

```python
import tensorflow as tf

# GPU testi
print("GPU Test:")
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0], [2.0]])
    c = tf.matmul(a, b)
    print(f"✅ GPU çalışıyor!")
    print(f"Sonuç: {c.numpy()}")
    print(f"İşlem cihazı: {c.device}")
```

### GPU Bilgilerini Görüntüleme

```python
import tensorflow as tf

# GPU detayları
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    print(f"GPU: {gpu}")
    try:
        details = tf.config.experimental.get_device_details(gpu)
        print(f"Detaylar: {details}")
    except:
        pass
```

### GPU Bellek Kullanımı

```python
import tensorflow as tf

# GPU bellek bilgisi
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        # Bellek büyümesi aktif mi?
        print(f"GPU: {gpu}")
        # TensorFlow otomatik olarak bellek yönetir
```

---

## ⚙️ GPU Optimizasyonu

### Bellek Büyümesi (Memory Growth)

```python
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Bellek büyümesi - sadece ihtiyaç duyduğu kadar bellek kullanır
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ Bellek büyümesi aktif")
    except RuntimeError as e:
        print(f"⚠️ Hata: {e}")
```

### Bellek Limiti (Opsiyonel)

```python
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # GPU belleğinin %50'sini kullan
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]  # 4GB
    )
```

### Mixed Precision (Hızlandırma)

```python
# config.py'de
GPU_CONFIG = {
    "use_gpu": True,
    "gpu_memory_growth": True,
    "mixed_precision": True,  # ✅ Hızlandırma için
}

# Eğitim sırasında
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')  # Hızlandırma
```

---

## 🐛 Sorun Giderme

### Problem: "GPU bulunamadı"

**Yerel Bilgisayar:**
```powershell
# NVIDIA driver kontrolü
nvidia-smi

# TensorFlow GPU versiyonu
pip install tensorflow[and-cuda]
```

**Google Colab:**
- Runtime → Change runtime type → GPU seçin
- Runtime → Restart runtime

### Problem: "CUDA out of memory"

**Çözüm:**
```python
# Batch size'ı küçült
MODEL_CONFIG = {
    "batch_size": 8,  # 16'dan 8'e düşür
}

# Veya bellek büyümesi aktif et
tf.config.experimental.set_memory_growth(gpu, True)
```

### Problem: "Model CPU'da çalışıyor"

**Çözüm:**
```python
# GPU'yu açıkça belirt
with tf.device('/GPU:0'):
    model = tf.keras.models.load_model("model.h5")
    predictions = model.predict(images)
```

---

## 📊 Performans Karşılaştırması

| İşlem | CPU | GPU |
|-------|-----|-----|
| Model Yükleme | ~5 saniye | ~2 saniye |
| 1 Batch Tahmin (32 görüntü) | ~3 saniye | ~0.1 saniye |
| 1 Epoch (3,889 görüntü) | ~15-20 dakika | ~2-3 dakika |
| 20 Epoch | ~5-7 saat | ~40-60 dakika |

**GPU kullanımı 10-20x daha hızlı!**

---

## ✅ Hızlı Başlangıç

### Yerel Bilgisayar:

```python
# 1. GPU kontrolü
python gpu_kontrol.py

# 2. Eğitimi başlat (otomatik GPU kullanır)
python train.py
```

### Google Colab:

```python
# 1. Runtime → Change runtime type → GPU

# 2. GPU kontrolü
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU: {len(gpus)}")

# 3. Eğitimi başlat
!python train.py
```

---

## 🎯 Özet

1. **GPU Kontrolü:** `python gpu_kontrol.py` veya `tf.config.list_physical_devices('GPU')`
2. **Config Ayarları:** `config.py` → `GPU_CONFIG["use_gpu"] = True`
3. **Eğitim:** `python train.py` (otomatik GPU kullanır)
4. **Model Yükleme:** Model otomatik olarak GPU'da çalışır

**Model GPU'da otomatik çalışır - ekstra kod gerekmez!** 🚀

