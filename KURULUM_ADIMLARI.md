# 📦 Kurulum Adımları - Adım Adım

Bu rehber, sıfırdan başlayanlar için hazırlanmıştır.

## 🎯 Şu Anda Yapmanız Gerekenler

### ADIM 1: Gerekli Paketleri Yükleyin

PowerShell veya Terminal'i açın ve şu komutu çalıştırın:

```bash
pip install -r requirements.txt
```

**Not**: Bu işlem 10-20 dakika sürebilir. İnternet bağlantınızın iyi olduğundan emin olun.

### ADIM 2: Kurulumu Kontrol Edin

Kurulum tamamlandıktan sonra:

```bash
python hizli_baslangic.py
```

Bu script, her şeyin doğru kurulup kurulmadığını kontrol eder.

### ADIM 3: Veri Setini Kontrol Edin

```bash
python check_data.py
```

Bu script, veri setinizin yapısını gösterir ve hazır olup olmadığını kontrol eder.

### ADIM 4: İlk Test Eğitimi (Küçük)

`config.py` dosyasını açın ve şu değişiklikleri yapın:

```python
MODEL_CONFIG = {
    "batch_size": 8,   # 32 yerine 8 (daha az RAM kullanır)
    "epochs": 3,       # 100 yerine 3 (hızlı test için)
    ...
}
```

Sonra eğitimi başlatın:

```bash
python train.py
```

Bu, küçük bir test çalıştırır ve her şeyin çalışıp çalışmadığını gösterir.

---

## ⚠️ Önemli Notlar

### Eğer "pip" komutu çalışmıyorsa:

1. Python'un yüklü olduğundan emin olun
2. `python -m pip install -r requirements.txt` komutunu deneyin
3. Veya `py -m pip install -r requirements.txt` komutunu deneyin

### Eğer "Out of Memory" hatası alırsanız:

`config.py` dosyasında batch_size'ı daha da küçültün:

```python
"batch_size": 4,  # veya 2
```

### Eğer GPU kullanmak istiyorsanız:

1. NVIDIA GPU'nuzun olduğundan emin olun
2. CUDA Toolkit'i yükleyin
3. TensorFlow GPU versiyonunu yükleyin:
   ```bash
   pip install tensorflow-gpu
   ```

---

## 📚 Daha Fazla Bilgi

Detaylı bilgi için `BASLANGIC_REHBERI.md` dosyasına bakın.

---

## 🆘 Sorun mu Yaşıyorsunuz?

1. Hata mesajını tam olarak okuyun
2. `BASLANGIC_REHBERI.md` dosyasındaki "Sorun Giderme" bölümüne bakın
3. Proje ekibi ile iletişime geçin

---

**Başarılar! 🚀**

