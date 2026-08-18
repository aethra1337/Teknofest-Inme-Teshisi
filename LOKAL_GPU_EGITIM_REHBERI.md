# 🖥️ Yerel Bilgisayarda GPU ile Eğitim Rehberi

## 🚀 Hızlı Başlangıç

### 1. GPU Kontrolü

```powershell
# GPU kontrolü
python gpu_kontrol.py
```

**Başarılı çıktı:**
```
✅ GPU bulundu: 1 adet
✅ GPU kullanılabilir ve yapılandırıldı!
```

### 2. Eğitimi Başlat

```powershell
# Sadece Akut için 20 epoch eğitim
python train_akut_20epoch_local.py
```

---

## 📋 Ön Gereksinimler

### GPU Kurulumu

1. **NVIDIA GPU Driver:**
   ```powershell
   nvidia-smi
   ```
   - Çalışmazsa: https://www.nvidia.com/Download/

2. **CUDA Toolkit:**
   - TensorFlow 2.16.1 için CUDA 12.x gerekli
   - https://developer.nvidia.com/cuda-downloads

3. **TensorFlow GPU:**
   ```powershell
   pip install tensorflow[and-cuda]
   ```

### Config Kontrolü

`config.py` dosyasında:

```python
GPU_CONFIG = {
    "use_gpu": True,  # ✅ True olmalı
    "gpu_memory_growth": True,
    "mixed_precision": False,
}
```

---

## 🎯 Eğitim Parametreleri

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| **Epochs** | 20 | Eğitim döngüsü |
| **Batch Size** | 32 | GPU için optimize |
| **Sınıflar** | Akut, İnme Yok | Sadece bu 2 sınıf |
| **Akut Ağırlık** | 3x | Sınıf dengesizliği için |
| **Learning Rate** | 0.0001 | Başlangıç öğrenme hızı |
| **Dropout** | 0.5 | Overfitting önleme |

---

## 📊 Beklenen Süre

| İşlem | Süre |
|-------|------|
| Veri hazırlama | ~5-10 dakika |
| 1 Epoch | ~5-10 dakika |
| 20 Epoch | ~1.5-3 saat |
| Test değerlendirme | ~1-2 dakika |

**Toplam: ~2-3.5 saat**

---

## 📁 Çıktı Dosyaları

Eğitim sonunda:

1. **Model:**
   - `models/saved_models/best_model_akut_20epoch.h5`

2. **Checkpoint:**
   - `models/checkpoints/[tarih]/best_model_akut_20epoch.h5`

3. **Eğitim Geçmişi:**
   - `results/training_history_akut_20epoch.json`

4. **Test Sonuçları:**
   - `results/classification_report.csv`
   - `results/metrics.csv`
   - `results/confusion_matrix.png`

---

## 🔍 GPU Kullanımını İzleme

### Terminal 1: Eğitim

```powershell
python train_akut_20epoch_local.py
```

### Terminal 2: GPU İzleme

```powershell
# Windows'ta
nvidia-smi -l 1

# Veya PowerShell'de
while ($true) { nvidia-smi; Start-Sleep -Seconds 2; Clear-Host }
```

**Göreceğiniz:**
- GPU kullanım yüzdesi
- Bellek kullanımı
- Sıcaklık

---

## 🐛 Sorun Giderme

### Problem: "GPU bulunamadı"

**Çözüm:**
```powershell
# 1. GPU kontrolü
nvidia-smi

# 2. TensorFlow GPU versiyonu
pip install tensorflow[and-cuda]

# 3. Test
python gpu_kontrol.py
```

### Problem: "CUDA out of memory"

**Çözüm:**
```python
# train_akut_20epoch_local.py içinde batch_size'ı küçült
training_config = {
    "batch_size": 16,  # 32'den 16'ya düşür
    ...
}
```

### Problem: "Model CPU'da çalışıyor"

**Çözüm:**
- `config.py` → `GPU_CONFIG["use_gpu"] = True`
- Runtime'ı yeniden başlatın

---

## 📈 Performans İpuçları

### 1. Batch Size Optimizasyonu

```python
# GPU belleğine göre ayarlayın
# 4GB GPU: batch_size = 8-16
# 8GB GPU: batch_size = 16-32
# 16GB+ GPU: batch_size = 32-64
```

### 2. Mixed Precision (Hızlandırma)

```python
# config.py'de
GPU_CONFIG = {
    "mixed_precision": True,  # ✅ Hızlandırma
}
```

### 3. Bellek Yönetimi

```python
# Eğitim sırasında gereksiz değişkenleri silin
del variable_name
```

---

## ✅ Başarı Kontrolü

Eğitim başarılıysa şunları görmelisiniz:

```
✅ 1 GPU bulundu ve yapılandırıldı
✅ Veri seti hazır!
✅ Model oluşturuldu
✅ Eğitim başlatılıyor...
Epoch 1/20
...
Epoch 20/20
✅ Eğitim tamamlandı!
✅ Model kaydedildi
🎉 TÜM İŞLEMLER TAMAMLANDI!
```

---

## 🎯 Sonraki Adımlar

1. **Modeli Test Et:**
   ```python
   from models.stroke_classifier import StrokeClassifier
   from pathlib import Path
   
   model = StrokeClassifier()
   model.load_model(Path("models/saved_models/best_model_akut_20epoch.h5"))
   ```

2. **Sonuçları Analiz Et:**
   - `results/classification_report.csv`
   - `results/confusion_matrix.png`

3. **TensorBoard ile İzle:**
   ```powershell
   tensorboard --logdir results/logs
   ```

---

## 💡 İpuçları

1. **Eğitim sırasında bilgisayarı kapatmayın**
2. **GPU sıcaklığını izleyin** (80°C altında tutun)
3. **Enerji tasarrufu modunu kapatın**
4. **Diğer GPU kullanan programları kapatın**

---

## 🆘 Yardım

Sorun yaşarsanız:

1. `python gpu_kontrol.py` çalıştırın
2. `nvidia-smi` ile GPU durumunu kontrol edin
3. Log dosyasını kontrol edin: `results/logs/training.log`

---

**Başarılar! 🚀**

