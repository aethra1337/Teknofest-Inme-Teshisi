# Sonraki Adımlar Rehberi

## 📊 Şu Anki Durum

### ✅ Tamamlananlar:
1. ✅ Veri seti analizi yapıldı (5,557 PNG görüntü)
2. ✅ Model eğitildi (EfficientNetB3)
3. ✅ Test değerlendirmesi yapıldı
4. ✅ Sonuçlar kaydedildi

### 📈 Mevcut Sonuçlar:
- **Test Accuracy**: %77.46
- **F1 Score**: 0.7343
- **İnme Yok Sınıfı**: %93.52 recall (çok iyi!)
- **Akut Sınıfı**: %14.71 recall (düşük - iyileştirilmeli)

---

## 🎯 Şimdi Ne Yapmalısınız?

### 1. MODEL İYİLEŞTİRME (Öncelikli)

Modeliniz çalışıyor ama **Akut sınıfını** yeterince iyi ayırt edemiyor. Bunu düzeltmek için:

#### A) Config Ayarlarını Güncelleyin

`config.py` dosyasında şu değişiklikleri yapın:

```python
MODEL_CONFIG = {
    "input_shape": (224, 224, 3),
    "num_classes": 2,
    "batch_size": 16,  # 8'den 16'ya artırın (daha hızlı eğitim)
    "epochs": 20,      # 3'ten 20'ye artırın (daha iyi öğrenme)
    "learning_rate": 0.0001,  # 0.001'den 0.0001'e düşürün (daha stabil)
    "dropout_rate": 0.7,  # 0.5'ten 0.7'ye artırın (overfitting önleme)
    ...
}
```

#### B) Yeniden Eğitim

```bash
python train.py
```

Bu işlem 1-2 saat sürebilir (20 epoch için).

---

### 2. SONUÇLARI İNCELEME

#### A) Confusion Matrix'i İnceleyin
```bash
# results/confusion_matrix.png dosyasını açın
```

Bu dosya hangi sınıfların karıştırıldığını gösterir.

#### B) Yanlış Tahminleri İnceleyin
```bash
# results/predictions.csv dosyasını Excel'de açın
# "true_label" ve "predicted_label" sütunlarını karşılaştırın
```

Hangi görüntülerin yanlış tahmin edildiğini görün.

---

### 3. YARIŞMA İÇİN HAZIRLIK

TEKNOFEST 2025 yarışması için:

#### A) Model Performansını Artırın
- F1 Score'u 0.80+ seviyesine çıkarın
- Her iki sınıf için de recall'u %70+ yapın

#### B) Dokümantasyon Hazırlayın
1. **Proje Raporu**: Metodoloji, sonuçlar, analiz
2. **Model Açıklaması**: Kullanılan mimari, hiperparametreler
3. **Sonuç Tabloları**: Metrikler, confusion matrix

#### C) Kod Kalitesi
- Kod yorumlarını kontrol edin
- README.md'yi güncelleyin
- Gereksiz dosyaları temizleyin

---

### 4. ALTERNATİF YAKLAŞIMLAR (İsteğe Bağlı)

Eğer sonuçlar yeterince iyi olmazsa:

#### A) Farklı Model Mimarileri
- ResNet50, DenseNet121 gibi başka modeller deneyin
- Ensemble (birden fazla model birleştirme) yapın

#### B) Veri Artırma (Data Augmentation)
- Daha agresif augmentation teknikleri
- Mixup, CutMix gibi gelişmiş teknikler

#### C) Focal Loss
- Dengesiz veri setleri için özel loss fonksiyonu
- Akut sınıfına daha fazla önem verir

---

## 📝 Pratik Adımlar (Sırayla)

### Adım 1: Config'i Güncelle (5 dakika)
```bash
# config.py dosyasını açın ve yukarıdaki değişiklikleri yapın
```

### Adım 2: Yeniden Eğitim (1-2 saat)
```bash
python train.py
```

### Adım 3: Sonuçları Kontrol Et (10 dakika)
```bash
python test_degerlendirme.py
python egitim_ozeti.py
```

### Adım 4: Sonuçları Analiz Et (30 dakika)
- Confusion matrix'i inceleyin
- Yanlış tahminleri analiz edin
- Metrikleri karşılaştırın

### Adım 5: Gerekirse Tekrar Dene
- Farklı hiperparametreler
- Farklı model mimarisi
- Daha fazla epoch

---

## 🎓 Öğrenme Kaynakları

### Model İyileştirme:
- **Overfitting**: Validation loss artıyorsa dropout'u artırın
- **Underfitting**: Model öğrenemiyorsa epoch sayısını artırın
- **Sınıf Dengesizliği**: Class weights kullanın (zaten kullanılıyor)

### Yarışma İçin:
- F1 Score en önemli metrik (yarışma kriteri)
- Her iki sınıf için de dengeli performans önemli
- Dokümantasyon ve kod kalitesi değerlendirilir

---

## ⚠️ Önemli Notlar

1. **Veri Seti**: Sadece PNG dosyaları kullanılıyor (DICOM atlandı)
2. **Sınıf Sayısı**: Şu an 2 sınıf (Akut, İnme Yok)
3. **Model**: EfficientNetB3 (11.7M parametre)
4. **Eğitim Süresi**: 20 epoch için ~1-2 saat (CPU'da)

---

## 🚀 Hızlı Başlangıç

En hızlı iyileştirme için:

1. `config.py`'yi açın
2. `epochs: 3` → `epochs: 20` yapın
3. `dropout_rate: 0.5` → `dropout_rate: 0.7` yapın
4. `python train.py` çalıştırın
5. Sonuçları bekleyin

---

## 📞 Sorularınız İçin

- Kod hataları: Log dosyalarını kontrol edin (`results/logs/training.log`)
- Model performansı: `results/metrics.csv` dosyasını inceleyin
- Görselleştirme: `results/confusion_matrix.png` dosyasını açın

---

**Son Güncelleme**: 28 Aralık 2025
**Durum**: Model eğitildi, test sonuçları alındı, iyileştirme aşaması

