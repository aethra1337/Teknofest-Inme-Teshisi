# GPU Kurulum Rehberi

## 🔍 Durum Tespiti

Sisteminizde GPU bulunamadı. GPU kullanmak için aşağıdaki adımları takip edin.

---

## 📋 Gereksinimler

### 1. NVIDIA GPU Kontrolü

Önce sisteminizde NVIDIA GPU olup olmadığını kontrol edin:

```powershell
nvidia-smi
```

**Eğer komut çalışmazsa:**
- NVIDIA GPU driver'ları yüklü değil
- veya NVIDIA GPU yok

**Eğer komut çalışırsa:**
- GPU modeli ve CUDA versiyonu görünecek
- Örnek çıktı:
```
NVIDIA-SMI 535.xx       Driver Version: 535.xx       CUDA Version: 12.2
```

---

## 🛠️ GPU Kurulum Adımları

### Adım 1: NVIDIA GPU Driver Kurulumu

1. **GPU Modelinizi Öğrenin:**
   - Windows'ta: `Win + X` → `Device Manager` → `Display adapters`
   - Veya: `nvidia-smi` komutunu çalıştırın

2. **Driver İndirin:**
   - https://www.nvidia.com/Download/index.aspx
   - GPU modelinizi seçin ve driver'ı indirin
   - Kurulumu yapın

3. **Kontrol:**
   ```powershell
   nvidia-smi
   ```

### Adım 2: CUDA Toolkit Kurulumu

TensorFlow 2.20.0 için **CUDA 12.3** gereklidir.

1. **CUDA Toolkit İndirin:**
   - https://developer.nvidia.com/cuda-downloads
   - Windows → x86_64 → 10/11 → exe (local)
   - İndirip kurun

2. **cuDNN Kurulumu (Önemli!):**
   - https://developer.nvidia.com/cudnn
   - Hesap oluşturmanız gerekebilir (ücretsiz)
   - CUDA 12.x için cuDNN indirin
   - ZIP dosyasını açın ve içeriği CUDA kurulum dizinine kopyalayın
   - Genellikle: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\`

3. **Kontrol:**
   ```powershell
   nvcc --version
   ```

### Adım 3: TensorFlow GPU Versiyonu

Mevcut TensorFlow CPU versiyonu kurulu. GPU versiyonunu kurmanız gerekiyor:

```powershell
# Mevcut TensorFlow'u kaldır
pip uninstall tensorflow

# GPU versiyonunu kur
pip install tensorflow[and-cuda]
```

**VEYA** (Manuel kurulum):

```powershell
pip install tensorflow==2.20.0
pip install nvidia-cudnn-cu12==8.9.7.29
```

### Adım 4: Kurulum Kontrolü

```powershell
python gpu_kontrol.py
```

**Başarılı olursa:**
```
[+] GPU bulundu!
[+] GPU kullanılabilir ve yapılandırıldı!
```

---

## ⚡ Hızlı Çözüm (Alternatif)

Eğer GPU kurulumu zor geliyorsa:

### Seçenek 1: Google Colab Kullanın
- Ücretsiz GPU erişimi
- Kurulum gerektirmez
- https://colab.research.google.com

### Seçenek 2: CPU ile Devam Edin
- Daha yavaş ama çalışır
- Epoch sayısını azaltın (3-5 epoch)
- Batch size'ı küçültün (4-8)

---

## 🔧 Sorun Giderme

### Problem: "CUDA out of memory"
**Çözüm:**
- Batch size'ı küçültün (8 → 4)
- `config.py`'de: `"batch_size": 4`

### Problem: "Could not load library cudnn"
**Çözüm:**
- cuDNN düzgün kurulmamış
- cuDNN dosyalarını CUDA dizinine kopyalayın

### Problem: "No GPU devices found"
**Çözüm:**
1. `nvidia-smi` çalışıyor mu kontrol edin
2. TensorFlow GPU versiyonu kurulu mu kontrol edin
3. CUDA versiyonu uyumlu mu kontrol edin

---

## 📊 Performans Karşılaştırması

| İşlem | CPU | GPU |
|-------|-----|-----|
| 1 Epoch (3,889 görüntü) | ~15-20 dakika | ~2-3 dakika |
| 20 Epoch | ~5-7 saat | ~40-60 dakika |
| Test (834 görüntü) | ~1 dakika | ~10 saniye |

**GPU kullanımı 10-20x daha hızlı!**

---

## ✅ Kurulum Sonrası

GPU kurulumu tamamlandıktan sonra:

1. **Config'i kontrol edin:**
   ```python
   GPU_CONFIG = {
       "use_gpu": True,  # True olmalı
       "gpu_memory_growth": True,
   }
   ```

2. **Eğitimi başlatın:**
   ```powershell
   python train.py
   ```

3. **GPU kullanımını kontrol edin:**
   ```powershell
   # Başka bir terminalde
   nvidia-smi
   ```

---

## 🆘 Yardım

- **NVIDIA Driver:** https://www.nvidia.com/Download/
- **CUDA Toolkit:** https://developer.nvidia.com/cuda-downloads
- **cuDNN:** https://developer.nvidia.com/cudnn
- **TensorFlow GPU:** https://www.tensorflow.org/install/gpu

---

**Not:** GPU kurulumu 30-60 dakika sürebilir. Sabırlı olun!

