# 🔧 CUDA PATH Ekleme Rehberi

## ⚠️ Sorun

CUDA PATH'te görünmüyor. Bu yüzden TensorFlow GPU'yu bulamıyor.

## ✅ Çözüm: CUDA'yı PATH'e Ekle

### Yöntem 1: Geçici (Bu Oturum İçin)

PowerShell'de şu komutları çalıştırın:

```powershell
# CUDA bin dizinini PATH'e ekle
$env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin"
$env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\libnvvp"

# Kontrol et
$env:PATH -split ';' | Select-String "CUDA"
```

**Sonra GPU kontrolü:**
```powershell
python gpu_kontrol.py
```

### Yöntem 2: Kalıcı (Önerilen)

1. **Windows tuşu + R** → `sysdm.cpl` yazın → Enter
2. **"Advanced"** sekmesine gidin
3. **"Environment Variables"** butonuna tıklayın
4. **"System variables"** altında **"Path"** seçin
5. **"Edit"** butonuna tıklayın
6. **"New"** butonuna tıklayın ve şu dizinleri ekleyin:

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\libnvvp
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\lib\x64
```

7. **OK** → **OK** → **OK**
8. **Tüm pencereleri kapatın ve yeniden açın**

---

## 🎯 Hızlı Adımlar (PowerShell'de)

### Adım 1: CUDA Dizinini Kontrol Et

```powershell
Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin"
```

**True dönerse:** CUDA kurulu ✅

### Adım 2: Geçici PATH Ekle

```powershell
$env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin"
```

### Adım 3: GPU Kontrolü

```powershell
python gpu_kontrol.py
```

---

## 📝 Tam Komut Seti (Kopyala-Yapıştır)

PowerShell'de sırayla çalıştırın:

```powershell
# 1. CUDA dizinini kontrol et
Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin"

# 2. PATH'e ekle (geçici)
$env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin"
$env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\libnvvp"

# 3. Kontrol et
$env:PATH -split ';' | Select-String "CUDA"

# 4. GPU kontrolü
python gpu_kontrol.py
```

---

## ⚠️ Önemli Notlar

1. **CUDA versiyonu farklıysa:** `v13.1` yerine kendi versiyonunuzu yazın
2. **CUDA versiyonunu öğrenmek için:**
   ```powershell
   Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\" | Select-Object Name
   ```
3. **Kalıcı ekleme yaptıysanız:** Tüm pencereleri kapatıp yeniden açın

---

## ✅ Başarı Kontrolü

PATH eklendikten sonra:

```powershell
python -c "import tensorflow as tf; print('GPU\'lar:', tf.config.list_physical_devices('GPU'))"
```

**Beklenen çıktı:**
```
GPU'lar: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

