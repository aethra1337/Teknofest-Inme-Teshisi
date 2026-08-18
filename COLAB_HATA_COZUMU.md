# 🔧 Colab Hata Çözümü: RecursionError

## Problem
```
RecursionError: maximum recursion depth exceeded
```

Bu hata, NumPy 1.26.4 ve TensorFlow 2.16.1 arasındaki uyumsuzluktan kaynaklanır.

## ✅ Çözüm

### Yöntem 1: Notebook'u Güncelleyin (Önerilen)

Notebook'taki kütüphane kurulum hücresini şu şekilde değiştirin:

```python
# Önce numpy'yi uyumlu versiyona düşür
!pip install -q "numpy<1.26"

# Sonra diğer kütüphaneleri kur
!pip install -q tensorflow==2.16.1 keras==3.0.5
!pip install -q pandas==2.2.0
!pip install -q opencv-python==4.9.0.80 Pillow==10.2.0
!pip install -q scikit-learn==1.4.0 pydicom==2.4.4
!pip install -q matplotlib==3.8.2 seaborn==0.13.2
```

### Yöntem 2: Manuel Düzeltme

Colab'de yeni bir hücre açın ve şunu çalıştırın:

```python
# Önce mevcut numpy'yi kaldır
!pip uninstall -y numpy

# Uyumlu versiyonu kur
!pip install -q "numpy<1.26"

# TensorFlow'u yeniden kur
!pip install -q --force-reinstall tensorflow==2.16.1

# Test et
import tensorflow as tf
print(f"✅ TensorFlow: {tf.__version__}")
import numpy as np
print(f"✅ NumPy: {np.__version__}")
```

### Yöntem 3: Alternatif Versiyonlar

Eğer hala sorun varsa, daha eski ama stabil versiyonları deneyin:

```python
!pip install -q "numpy==1.24.3"
!pip install -q tensorflow==2.15.0
!pip install -q keras==2.15.0
```

---

## 🎯 Hızlı Çözüm (Tek Satır)

Colab'de yeni bir hücre açın:

```python
!pip uninstall -y numpy && pip install -q "numpy<1.26" && pip install -q --force-reinstall tensorflow==2.16.1
```

Sonra notebook'unuzu tekrar çalıştırın.

---

## 📝 Notlar

- NumPy 1.26+ versiyonları TensorFlow 2.16 ile uyumsuz olabilir
- NumPy 1.24 veya 1.25 versiyonları daha stabil
- TensorFlow 2.16.1, NumPy <1.26 ile çalışır

---

**Sorun devam ederse:** Runtime → Restart runtime yapın ve tekrar deneyin.

