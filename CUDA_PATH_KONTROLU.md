# 🔍 CUDA PATH Kontrolü

## 📋 Adımlar

### 1. PowerShell Terminali Açın

- **Cursor'da:** Alt + ` (backtick) tuşu ile terminal açın
- **Veya:** Windows tuşu + X → "Windows PowerShell" veya "Terminal"

### 2. Proje Klasörüne Gidin

```powershell
cd "C:\Users\LENOVO\Desktop\Projeler\Teknofest"
```

### 3. CUDA PATH Kontrolü

Komutu yapıştırın ve Enter'a basın:

```powershell
$env:PATH -split ';' | Select-String "CUDA"
```

### 4. Sonuç

**Eğer CUDA görünürse:**
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\libnvvp
```

**Eğer hiçbir şey görünmezse:**
- CUDA PATH'e eklenmemiş demektir
- Manuel olarak eklemeniz gerekebilir

---

## 🖼️ Görsel Rehber

1. **Cursor'da Terminal Aç:**
   - Alt + ` tuşuna basın
   - Veya: View → Terminal

2. **Komutu Yapıştır:**
   ```
   $env:PATH -split ';' | Select-String "CUDA"
   ```

3. **Enter'a Bas**

---

## ⚠️ Eğer CUDA Görünmezse

Manuel PATH ekleme:

```powershell
# Geçici olarak (sadece bu oturum için)
$env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin"
$env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\libnvvp"
```

**Kalıcı olarak eklemek için:**
1. Windows tuşu → "Environment Variables" ara
2. "Edit the system environment variables" aç
3. "Environment Variables" butonuna tıkla
4. "Path" → "Edit"
5. CUDA dizinlerini ekle

---

## ✅ Hızlı Test

PATH kontrolünden sonra:

```powershell
python gpu_kontrol.py
```

GPU görünmeli!

