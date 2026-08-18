# 📦 Google Colab için Gereksinimler (Requirements)

## 🚀 Hızlı Kurulum

Google Colab'de tek bir hücrede tüm kütüphaneleri kurmak için:

```python
# NumPy uyumsuzluğu düzelt (ÖNEMLİ!)
!pip uninstall -y numpy

# Tüm kütüphaneleri kur
!pip install -q "numpy<1.26"
!pip install -q --force-reinstall tensorflow==2.16.1
!pip install -q keras==3.0.5
!pip install -q pandas==2.2.0
!pip install -q opencv-python==4.9.0.80 Pillow==10.2.0
!pip install -q scikit-learn==1.4.0 pydicom==2.4.4
!pip install -q matplotlib==3.8.2 seaborn==0.13.2

print("✅ Tüm kütüphaneler kuruldu!")
```

## 📋 Detaylı Kütüphane Listesi

### 🔴 Kritik (Mutlaka Gerekli)

| Kütüphane | Versiyon | Açıklama |
|-----------|----------|----------|
| **numpy** | `<1.26` | ⚠️ TensorFlow uyumluluğu için kritik! |
| **tensorflow** | `2.16.1` | Deep learning framework |
| **keras** | `3.0.5` | Yüksek seviye API |

### 🟡 Önemli (Proje için Gerekli)

| Kütüphane | Versiyon | Açıklama |
|-----------|----------|----------|
| **pandas** | `2.2.0` | Veri işleme ve analiz |
| **opencv-python** | `4.9.0.80` | Görüntü işleme |
| **Pillow** | `10.2.0` | Görüntü manipülasyonu |
| **scikit-learn** | `1.4.0` | Makine öğrenmesi araçları |
| **pydicom** | `2.4.4` | DICOM dosya okuma |

### 🟢 Görselleştirme (Opsiyonel ama Önerilen)

| Kütüphane | Versiyon | Açıklama |
|-----------|----------|----------|
| **matplotlib** | `3.8.2` | Grafik çizimi |
| **seaborn** | `0.13.2` | İstatistiksel görselleştirme |
| **tensorboard** | `>=2.15.0` | Model eğitim izleme |

## ⚠️ Önemli Notlar

### 1. NumPy Versiyonu

**ÇOK ÖNEMLİ**: NumPy 1.26+ versiyonları TensorFlow ile uyumsuzdur ve `RecursionError` hatası verir!

```python
# ❌ YANLIŞ
!pip install numpy

# ✅ DOĞRU
!pip uninstall -y numpy
!pip install "numpy<1.26"
```

### 2. Kurulum Sırası

Kütüphaneleri şu sırayla kurun:

1. Önce NumPy'yi kaldırın
2. Uyumlu NumPy versiyonunu kurun
3. TensorFlow'u kurun
4. Diğer kütüphaneleri kurun

### 3. Versiyon Kontrolü

Kurulumdan sonra versiyonları kontrol edin:

```python
import numpy as np
import tensorflow as tf
import pandas as pd

print(f"NumPy: {np.__version__}")
print(f"TensorFlow: {tf.__version__}")
print(f"Pandas: {pd.__version__}")

# NumPy 1.26'dan küçük olmalı!
assert int(np.__version__.split('.')[1]) < 26, "NumPy versiyonu çok yüksek!"
```

## 🔧 Sorun Giderme

### Hata: `RecursionError: maximum recursion depth exceeded`

**Çözüm:**
```python
!pip uninstall -y numpy
!pip install "numpy<1.26"
# Runtime'ı yeniden başlatın
```

### Hata: `CUDA out of memory`

**Çözüm:**
- Batch size'ı azaltın (32 → 16)
- Runtime'ı yeniden başlatın

### Hata: `ModuleNotFoundError`

**Çözüm:**
```python
# Eksik kütüphaneyi kurun
!pip install [kütüphane_adı]
```

## 📝 Requirements Dosyası Kullanımı

Eğer `requirements_colab.txt` dosyanız varsa:

```python
# Dosyayı yükleyin (Google Drive'dan veya direkt)
!pip install -r requirements_colab.txt
```

## 🎯 Minimal Kurulum (Sadece Temel)

Eğer sadece temel işlevler için:

```python
!pip uninstall -y numpy
!pip install -q "numpy<1.26"
!pip install -q tensorflow==2.16.1 keras==3.0.5
!pip install -q pandas opencv-python Pillow scikit-learn
```

## 📦 Tam Kurulum (Tüm Özellikler)

Tüm özellikler için (görselleştirme dahil):

```python
!pip uninstall -y numpy
!pip install -q "numpy<1.26"
!pip install -q --force-reinstall tensorflow==2.16.1
!pip install -q keras==3.0.5
!pip install -q pandas==2.2.0
!pip install -q opencv-python==4.9.0.80 Pillow==10.2.0
!pip install -q scikit-learn==1.4.0 pydicom==2.4.4
!pip install -q matplotlib==3.8.2 seaborn==0.13.2
!pip install -q tensorboard
```

## ✅ Kurulum Doğrulama

Kurulumun başarılı olduğunu kontrol edin:

```python
try:
    import numpy as np
    import tensorflow as tf
    import keras
    import pandas as pd
    import cv2
    import PIL
    import sklearn
    import pydicom
    import matplotlib
    import seaborn
    
    print("✅ Tüm kütüphaneler başarıyla yüklendi!")
    print(f"NumPy: {np.__version__}")
    print(f"TensorFlow: {tf.__version__}")
    print(f"Keras: {keras.__version__}")
except ImportError as e:
    print(f"❌ Hata: {e}")
```

## 🔄 Runtime Yeniden Başlatma

Kütüphane kurulumundan sonra:

1. **Runtime → Restart runtime** (Önerilen)
2. Veya kod içinde: `import os; os.kill(os.getpid(), 9)`

## 📚 Ek Kaynaklar

- [TensorFlow Kurulum Rehberi](https://www.tensorflow.org/install)
- [NumPy Versiyon Uyumluluğu](https://numpy.org/doc/stable/)
- [Google Colab GPU Kullanımı](https://colab.research.google.com/notebooks/gpu.ipynb)

---

**Not**: Bu requirements listesi Google Colab ortamı için optimize edilmiştir. Yerel ortam için `requirements.txt` dosyasını kullanın.

