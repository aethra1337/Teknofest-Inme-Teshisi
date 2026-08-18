# 🚀 TEKNOFEST - Akut Sınıfı İyileştirme Rehberi

## 📋 Özet

Bu rehber, **Akut İnme** sınıfının performansını iyileştirmek için Google Colab'de 25 epoch eğitim yapmanızı sağlar.

## 🎯 İyileştirmeler

1. **Sınıf Ağırlıkları**: Akut sınıfı için 3x ağırlık artırımı
2. **Geliştirilmiş Augmentation**: Daha agresif veri artırma
3. **25 Epoch**: Daha uzun eğitim
4. **GPU Optimizasyonu**: Batch size artırıldı (32)
5. **Dropout Azaltma**: 0.7 → 0.5 (daha az overfitting riski)

## 📁 Dosya Yapısı

Google Drive'ınızda şu yapı olmalı:
```
MyDrive/
└── Teknofest/
    ├── iskemi/
    ├── inmeyok/
    ├── config.py
    ├── train.py
    ├── utils/
    ├── models/
    ├── training/
    └── evaluation/
```

## 🚀 Adım Adım Kurulum

### 1. Google Colab'de Yeni Notebook Oluşturun

1. [Google Colab](https://colab.research.google.com/) açın
2. **File → New notebook** seçin
3. **Runtime → Change runtime type → GPU** seçin

### 2. İlk Hücre: Drive Bağlantısı ve Kurulum

```python
# Google Drive'ı bağla
from google.colab import drive
drive.mount('/content/drive')

# Proje yolunu ayarla (KENDİ YOLUNUZA GÖRE DEĞİŞTİRİN!)
import os
PROJECT_PATH = "/content/drive/MyDrive/Teknofest"  # ⚠️ DEĞİŞTİRİN!
os.chdir(PROJECT_PATH)
print(f"✅ Proje dizini: {os.getcwd()}")

# Kütüphaneleri kur
!pip uninstall -y numpy
!pip install -q "numpy<1.26"
!pip install -q --force-reinstall tensorflow==2.16.1
!pip install -q keras==3.0.5 pandas==2.2.0
!pip install -q opencv-python==4.9.0.80 Pillow==10.2.0
!pip install -q scikit-learn==1.4.0 pydicom==2.4.4
!pip install -q matplotlib==3.8.2 seaborn==0.13.2

print("✅ Kurulum tamamlandı!")
```

### 3. İkinci Hücre: GPU Kontrolü

```python
import tensorflow as tf
import numpy as np

print(f"TensorFlow: {tf.__version__}")
print(f"NumPy: {np.__version__}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU bulundu: {len(gpus)}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("❌ GPU bulunamadı!")
```

### 4. Üçüncü Hücre: Tüm Eğitim Kodu

`colab_train_akut_iyilestirme.py` dosyasının içeriğini buraya kopyalayın veya aşağıdaki kodu kullanın:

```python
# Tüm eğitim kodu buraya gelecek
# colab_train_akut_iyilestirme.py dosyasının içeriği
```

## 📊 Beklenen Sonuçlar

### Önceki Performans (3 epoch):
- **Akut F1-Score**: %21.01
- **Genel Accuracy**: %77.46

### Hedef Performans (25 epoch + iyileştirmeler):
- **Akut F1-Score**: %50+ (hedef)
- **Genel Accuracy**: %80+ (hedef)

## ⚙️ Önemli Parametreler

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| Epochs | 25 | Eğitim döngüsü sayısı |
| Batch Size | 32 | GPU için optimize edildi |
| Learning Rate | 0.0001 | Başlangıç öğrenme hızı |
| Dropout | 0.5 | Overfitting önleme |
| Akut Ağırlık | 3x | Sınıf dengesizliği için |

## 🔧 Özelleştirme

### Sınıf Ağırlığını Değiştirme

Kodda şu satırı bulun:
```python
if class_name == "Akut":
    weight = base_weight * 3.0  # Bu değeri değiştirin
```

- `2.0`: Daha az ağırlık
- `3.0`: Mevcut (önerilen)
- `4.0`: Daha fazla ağırlık

### Augmentation Parametrelerini Değiştirme

```python
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=30,  # Döndürme açısı
    width_shift_range=0.2,  # Yatay kaydırma
    height_shift_range=0.2,  # Dikey kaydırma
    # ... diğer parametreler
)
```

## 📈 İlerlemeyi İzleme

### TensorBoard ile

```python
# Eğitim sırasında başka bir hücrede:
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/Teknofest/results/logs
```

### Konsol Çıktısı

Her epoch sonunda şunları göreceksiniz:
- Loss (eğitim ve doğrulama)
- Accuracy
- Precision, Recall, F1-Score
- Learning rate (azalırsa)

## ⏱️ Süre Tahmini

- **25 Epoch**: 2-4 saat (GPU'ya bağlı)
- **Her Epoch**: ~5-10 dakika

## 💾 Model Kayıtları

Eğitim sonunda şu dosyalar oluşacak:

1. **En İyi Model**: `models/saved_models/best_model_akut_iyilestirme.h5`
2. **Checkpoint**: `models/checkpoints/[tarih]/best_model_akut_iyilestirme.h5`
3. **Eğitim Geçmişi**: `results/training_history_akut_iyilestirme.json`
4. **Test Sonuçları**: `results/classification_report.csv`

## 🐛 Sorun Giderme

### GPU Hatası
```
RuntimeError: CUDA out of memory
```
**Çözüm**: Batch size'ı 32'den 16'ya düşürün

### NumPy Hatası
```
RecursionError: maximum recursion depth exceeded
```
**Çözüm**: İlk hücrede `!pip uninstall -y numpy` çalıştırın

### Drive Bağlantı Hatası
```
MountError: Already mounted
```
**Çözüm**: Runtime'ı yeniden başlatın (Runtime → Restart runtime)

## 📞 Destek

Sorun yaşarsanız:
1. Konsol çıktısını kontrol edin
2. TensorBoard'da grafikleri inceleyin
3. Model checkpoint'lerini kontrol edin

## 🎉 Başarılar!

Eğitim tamamlandığında, Akut sınıfı performansında önemli bir iyileşme görmelisiniz!

