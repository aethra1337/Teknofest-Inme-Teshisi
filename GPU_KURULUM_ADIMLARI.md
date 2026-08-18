# 🚀 GPU Kurulum Adımları (RTX 3060 için)

## ✅ Mevcut Durum

- ✅ **GPU Bulundu:** NVIDIA GeForce RTX 3060
- ✅ **Driver:** 591.59 (Güncel)
- ✅ **CUDA:** 13.1 (Kurulu)
- ❌ **TensorFlow GPU:** Bulunamıyor

## 🔧 Çözüm: TensorFlow GPU Versiyonunu Kur

### Adım 1: Mevcut TensorFlow'u Kaldır

```powershell
pip uninstall tensorflow tensorflow-cpu -y
```

### Adım 2: TensorFlow GPU Versiyonunu Kur

TensorFlow 2.20.0 için CUDA 13.1 uyumlu versiyonu kurun:

```powershell
# TensorFlow GPU versiyonu
pip install tensorflow[and-cuda]
```

**VEYA** (Manuel kurulum - daha güvenilir):

```powershell
pip install tensorflow==2.20.0
pip install nvidia-cudnn-cu12==8.9.7.29
```

### Adım 3: Kurulumu Kontrol Et

```powershell
python gpu_kontrol.py
```

**Başarılı çıktı:**
```
✅ GPU bulundu: 1 adet
✅ GPU kullanılabilir ve yapılandırıldı!
```

---

## 🎯 Hızlı Kurulum Scripti

Aşağıdaki komutları sırayla çalıştırın:

```powershell
# 1. Mevcut TensorFlow'u kaldır
pip uninstall tensorflow tensorflow-cpu -y

# 2. GPU versiyonunu kur
pip install tensorflow[and-cuda]

# 3. Kontrol et
python gpu_kontrol.py
```

---

## ⚠️ Önemli Notlar

1. **CUDA 13.1:** TensorFlow 2.20.0 CUDA 13.1 ile uyumludur
2. **cuDNN:** Otomatik olarak kurulacak (`tensorflow[and-cuda]` ile)
3. **Yeniden Başlatma:** Kurulumdan sonra Python'u yeniden başlatın

---

## 🔍 Sorun Giderme

### Problem: "Could not load library cudnn"

**Çözüm:**
```powershell
pip install nvidia-cudnn-cu12==8.9.7.29
```

### Problem: "No GPU devices found"

**Çözüm:**
1. Python'u kapatıp yeniden açın
2. `python gpu_kontrol.py` çalıştırın
3. Hala çalışmazsa: `pip install --upgrade tensorflow[and-cuda]`

### Problem: "CUDA version mismatch"

**Çözüm:**
- CUDA 13.1 kurulu, TensorFlow 2.20.0 bunu destekler
- Eğer sorun devam ederse: `pip install tensorflow==2.16.1` deneyin

---

## ✅ Kurulum Sonrası

Kurulum başarılı olduktan sonra:

```powershell
# Eğitimi başlat
python train_akut_20epoch_local.py
```

---

## 📊 Beklenen Sonuç

GPU kurulumu başarılı olduğunda:

```
✅ GPU bulundu: 1 adet
  GPU 0: PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')
✅ GPU bellek büyümesi aktif
✅ GPU kullanılabilir ve yapılandırıldı!
```

