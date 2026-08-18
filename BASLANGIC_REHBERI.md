# 🚀 Başlangıç Rehberi - Model Geliştirme

Bu rehber, model geliştirme konusunda yeni olanlar için adım adım talimatlar içerir.

## 📋 İçindekiler

1. [Gereksinimler](#1-gereksinimler)
2. [Kurulum](#2-kurulum)
3. [Veri Setini Anlama](#3-veri-setini-anlama)
4. [İlk Test](#4-ilk-test)
5. [Model Eğitimi](#5-model-eğitimi)
6. [Sonuçları İnceleme](#6-sonuçları-inceleme)
7. [Sorun Giderme](#7-sorun-giderme)

---

## 1. Gereksinimler

### Donanım
- **RAM**: En az 8GB (16GB önerilir)
- **GPU**: NVIDIA GPU (CUDA destekli) - Opsiyonel ama önerilir
- **Disk Alanı**: En az 10GB boş alan

### Yazılım
- **Python**: 3.8 veya üzeri
- **İşletim Sistemi**: Windows, Linux veya Mac

---

## 2. Kurulum

### Adım 1: Python'u Kontrol Edin

Terminal/PowerShell'de şu komutu çalıştırın:

```bash
python --version
```

Eğer Python yüklü değilse, [python.org](https://www.python.org/downloads/) adresinden indirin.

### Adım 2: Sanal Ortam Oluşturun (Önerilir)

```bash
# Windows PowerShell
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### Adım 3: Gerekli Kütüphaneleri Yükleyin

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Not**: Bu işlem 5-10 dakika sürebilir. İlk kez yüklüyorsanız daha uzun sürebilir.

### Adım 4: GPU Desteği (Opsiyonel)

Eğer NVIDIA GPU'nuz varsa ve CUDA kullanmak istiyorsanız:

1. [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) indirin
2. TensorFlow GPU versiyonunu yükleyin:
   ```bash
   pip install tensorflow-gpu
   ```

---

## 3. Veri Setini Anlama

### Veri Seti Yapısı

Projenizde iki ana veri seti var:

1. **İskemi Veri Seti** (`iskemi/iskemi/İskemi Veri Seti/`)
   - DICOM klasörü: 1130 DICOM dosyası
   - PNG klasörü: 1130 PNG görüntüsü
   - OVERLAY klasörü: 1130 overlay görüntüsü

2. **İnme Yok Veri Seti** (`inmeyok/`)
   - PNG klasörü: 4427 PNG görüntüsü

### Veri Setini Kontrol Edin

Veri setinizi kontrol etmek için:

```bash
python check_data.py
```

Bu script:
- Tüm görüntüleri bulur
- Sınıf dağılımını gösterir
- Veri setini train/val/test olarak böler
- Sınıf ağırlıklarını hesaplar

---

## 4. İlk Test

### Adım 1: Yapılandırmayı Kontrol Edin

`config.py` dosyasını açın ve şunları kontrol edin:

- ✅ Veri seti yolları doğru mu?
- ✅ Model ayarları uygun mu? (Başlangıç için varsayılanlar yeterli)

### Adım 2: Küçük Bir Test Çalıştırın

İlk test için model ayarlarını küçültün. `config.py` dosyasında:

```python
MODEL_CONFIG = {
    "batch_size": 16,  # 32 yerine 16 (daha az RAM kullanır)
    "epochs": 5,        # 100 yerine 5 (hızlı test için)
    ...
}
```

### Adım 3: Veri Setini Kontrol Edin

```bash
python check_data.py
```

Eğer hata alırsanız, hata mesajını not edin ve [Sorun Giderme](#7-sorun-giderme) bölümüne bakın.

---

## 5. Model Eğitimi

### Adım 1: Eğitimi Başlatın

```bash
python train.py
```

### Ne Olacak?

1. **Veri Yükleme**: Tüm görüntüler yüklenir ve organize edilir
2. **Veri Bölümleme**: Veri seti %70 train, %15 validation, %15 test olarak bölünür
3. **Model Oluşturma**: Transfer learning modeli oluşturulur
4. **Eğitim**: Model eğitilmeye başlar
5. **Kayıt**: En iyi model otomatik olarak kaydedilir

### Eğitim Süresi

- **CPU ile**: 5-10 saat (veri seti boyutuna göre)
- **GPU ile**: 1-3 saat (GPU'ya göre değişir)

### Eğitim Sırasında

- Her epoch sonunda metrikler gösterilir
- En iyi model otomatik kaydedilir
- TensorBoard logları oluşturulur
- Eğer model iyileşmiyorsa, early stopping devreye girer

---

## 6. Sonuçları İnceleme

### Eğitim Tamamlandıktan Sonra

Sonuçlar `results/` klasöründe bulunur:

1. **metrics.csv**: Genel metrikler
   - Accuracy (Doğruluk)
   - Precision (Kesinlik)
   - Recall (Duyarlılık)
   - **F1 Score** (Ana metrik - TEKNOFEST için önemli)

2. **classification_report.csv**: Her sınıf için detaylı rapor

3. **confusion_matrix.png**: Karışıklık matrisi görselleştirmesi

4. **predictions.csv**: Tüm tahminler ve olasılıklar

### Model Dosyaları

- `models/saved_models/best_model.h5`: En iyi model
- `models/checkpoints/`: Eğitim checkpoint'leri

### TensorBoard ile Görselleştirme

```bash
tensorboard --logdir=results/logs
```

Tarayıcınızda `http://localhost:6006` adresini açın.

---

## 7. Sorun Giderme

### Problem: "ModuleNotFoundError"

**Çözüm**: Gerekli kütüphaneleri yükleyin
```bash
pip install -r requirements.txt
```

### Problem: "Out of Memory" Hatası

**Çözüm**: Batch size'ı küçültün (`config.py` içinde)
```python
MODEL_CONFIG = {
    "batch_size": 8,  # 32 yerine 8
    ...
}
```

### Problem: Veri Seti Bulunamıyor

**Çözüm**: `config.py` dosyasındaki yolları kontrol edin
```python
ISKEMI_DIR = RAW_DATA_DIR / "iskemi" / "iskemi" / "İskemi Veri Seti"
INMEYOK_DIR = RAW_DATA_DIR / "inmeyok" / "İnme Yok_kronik süreç_diğer Veri Set_PNG" / "İnme Yok_kronik süreç_diğer Veri Set_PNG"
```

### Problem: GPU Kullanılmıyor

**Çözüm**: 
1. CUDA ve cuDNN'in yüklü olduğundan emin olun
2. TensorFlow GPU versiyonunu yükleyin:
   ```bash
   pip install tensorflow-gpu
   ```

### Problem: Eğitim Çok Yavaş

**Çözüm**:
- GPU kullanın
- Batch size'ı artırın (RAM yeterliyse)
- Daha küçük bir model kullanın (EfficientNetB0 gibi)

### Problem: Model İyi Performans Göstermiyor

**Çözüm**:
- Daha fazla epoch eğitin
- Data augmentation ayarlarını değiştirin
- Farklı bir base model deneyin (ResNet50, VGG16, vb.)
- Learning rate'i ayarlayın

---

## 8. İpuçları ve Öneriler

### Başlangıç İçin

1. **Küçük Başlayın**: İlk test için 5-10 epoch yeterli
2. **İzleyin**: TensorBoard ile eğitimi izleyin
3. **Kaydedin**: Her denemeyi farklı bir isimle kaydedin
4. **Dokümante Edin**: Hangi ayarlarla ne sonuç aldığınızı not edin

### Model İyileştirme

1. **Hyperparameter Tuning**: Learning rate, batch size, dropout rate deneyin
2. **Data Augmentation**: Daha fazla augmentation ekleyin
3. **Transfer Learning**: Farklı pre-trained modeller deneyin
4. **Ensemble**: Birden fazla modeli birleştirin

### Performans Optimizasyonu

1. **Mixed Precision**: GPU'da hızlandırma için
2. **Model Quantization**: Model boyutunu küçültmek için
3. **Pruning**: Gereksiz parametreleri kaldırmak için

---

## 9. Sonraki Adımlar

Eğitim başarılı olduktan sonra:

1. ✅ Model performansını değerlendirin
2. ✅ F1 Score'u kontrol edin (TEKNOFEST için kritik)
3. ✅ Confusion matrix'i inceleyin
4. ✅ Hangi sınıflarda hata yapıldığını analiz edin
5. ✅ Modeli iyileştirmek için ayarlamalar yapın
6. ✅ Test seti üzerinde final değerlendirme yapın

---

## 📞 Yardım

Sorun yaşarsanız:

1. Hata mesajını tam olarak okuyun
2. Bu rehberdeki "Sorun Giderme" bölümüne bakın
3. TensorFlow/Keras dokümantasyonuna bakın
4. Proje ekibi ile iletişime geçin

---

**Başarılar! 🎉**

