# TEKNOFEST 2025 - Nöroloji Analitiği: İnme Teşhisi ve İnmenin Zamansal Sınıflandırılması

Bu proje, TEKNOFEST 2025 Sağlıkta Yapay Zeka Yarışması kapsamında geliştirilmiştir. Difüzyon MR görüntülerinin bilgisayarlı analizi ile iskemik inmenin zamansal evrelerinin (hiperakut, akut, subakut, kronik) sınıflandırılması hedeflenmektedir.

## 📋 Proje Özeti

Proje, nöroradyolojik görüntüleme tekniklerinin yapay zeka tabanlı otomasyonu yoluyla tedavi protokollerinin optimizasyonuna katkı sağlamayı amaçlamaktadır. Kontrastsız BT ve difüzyon MR (T2A, DWI, ADC) kesitlerinden oluşan multidisipliner bir veri seti kullanılarak zamansal sınıflandırma gerçekleştirilmektedir.

### Özellikler

- ✅ **Çoklu Görüntü Formatı Desteği**: DICOM ve PNG formatlarını destekler
- ✅ **Transfer Learning**: EfficientNet, ResNet, VGG, DenseNet gibi pre-trained modeller
- ✅ **F1 Score Odaklı Değerlendirme**: TEKNOFEST gereksinimlerine uygun metrikler
- ✅ **Zamansal Sınıflandırma**: Hiperakut, Akut, Subakut, Kronik, İnme Yok
- ✅ **Modüler Yapı**: Kolay genişletilebilir ve bakım yapılabilir kod yapısı
- ✅ **Kapsamlı Değerlendirme**: Confusion matrix, classification report, metrikler

## 🏗️ Proje Yapısı

```
Teknofest/
├── data/                    # İşlenmiş veri setleri
│   ├── processed/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
├── models/                  # Model dosyaları
│   ├── checkpoints/         # Eğitim checkpoint'leri
│   └── saved_models/        # Kaydedilmiş modeller
├── training/                # Eğitim modülleri
│   ├── trainer.py
│   └── __init__.py
├── evaluation/              # Değerlendirme modülleri
│   ├── evaluator.py
│   └── __init__.py
├── utils/                   # Yardımcı fonksiyonlar
│   ├── data_loader.py       # Veri yükleme
│   ├── image_preprocessing.py  # Görüntü ön işleme
│   ├── dicom_reader.py      # DICOM okuma
│   └── __init__.py
├── results/                 # Sonuçlar ve loglar
│   ├── logs/
│   └── predictions/
├── config.py                # Yapılandırma dosyası
├── train.py                 # Ana eğitim scripti
├── requirements.txt         # Python bağımlılıkları
└── README.md                # Bu dosya
```

## 🚀 Kurulum

### Gereksinimler

- Python 3.8+
- CUDA destekli GPU (önerilir)
- TensorFlow 2.15+
- Diğer bağımlılıklar (requirements.txt)

### Adımlar

1. **Repository'yi klonlayın veya indirin**

2. **Sanal ortam oluşturun (önerilir)**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows
```

3. **Bağımlılıkları yükleyin**
```bash
pip install -r requirements.txt
```

4. **Yapılandırmayı kontrol edin**
   - `config.py` dosyasını açın ve veri seti yollarını kontrol edin
   - Model hiperparametrelerini ihtiyacınıza göre ayarlayın

5. **Veri setini hazırlayın**
   - Veri setlerinizi `iskemi/` ve `inmeyok/` klasörlerine yerleştirin
   - DICOM veya PNG formatında olabilir

## 📊 Kullanım

### Eğitim

Ana eğitim scriptini çalıştırın:

```bash
python train.py
```

Eğitim sırasında:
- Veri seti otomatik olarak train/val/test olarak bölünür
- Model eğitilir ve checkpoint'ler kaydedilir
- TensorBoard logları oluşturulur
- En iyi model otomatik olarak kaydedilir

### Yapılandırma

`config.py` dosyasından aşağıdaki ayarları yapabilirsiniz:

- **Model hiperparametreleri**: Batch size, learning rate, epochs, vb.
- **Veri bölümleme**: Train/val/test oranları
- **Görüntü ön işleme**: Boyutlandırma, normalizasyon, augmentation
- **Transfer learning**: Base model seçimi (EfficientNetB3, ResNet50, vb.)
- **GPU ayarları**: Memory growth, mixed precision

### Model Seçimi

`config.py` içinde `TRANSFER_LEARNING` bölümünden base model seçebilirsiniz:

```python
TRANSFER_LEARNING = {
    "use_transfer_learning": True,
    "base_model": "EfficientNetB3",  # EfficientNetB3, ResNet50, VGG16, DenseNet121
    "trainable_layers": -1,  # -1: tüm katmanlar, 0: sadece classifier
    "weights": "imagenet",
}
```

## 📈 Değerlendirme

Eğitim tamamlandıktan sonra, sonuçlar `results/` klasöründe bulunur:

- `metrics.csv`: Genel metrikler (Accuracy, Precision, Recall, F1 Score)
- `classification_report.csv`: Sınıf bazlı detaylı rapor
- `confusion_matrix.png`: Confusion matrix görselleştirmesi
- `predictions.csv`: Tüm tahminler ve olasılıklar

### Metrikler

Proje, TEKNOFEST gereksinimlerine uygun olarak **F1 Score** metriklerini önceliklendirir:

- **Accuracy**: Genel doğruluk
- **Precision**: Kesinlik
- **Recall**: Duyarlılık
- **F1 Score**: Precision ve Recall'un harmonik ortalaması (Ana metrik)
- **Per-class metrics**: Her sınıf için ayrı metrikler

## 🔬 Model Mimarisi

### Transfer Learning Yaklaşımı

Proje, pre-trained modeller kullanarak transfer learning uygular:

1. **Base Model**: ImageNet üzerinde eğitilmiş bir model (EfficientNetB3, ResNet50, vb.)
2. **Feature Extraction**: Base model'in özellik çıkarıcı katmanları
3. **Classifier**: Özel sınıflandırıcı katmanları (Dense + Dropout)
4. **Output**: 5 sınıflı softmax çıkışı

### Özellikler

- **Data Augmentation**: Rotation, translation, zoom, flip
- **Batch Normalization**: Eğitim stabilitesi için
- **Dropout**: Overfitting önleme
- **Early Stopping**: Overfitting önleme
- **Learning Rate Reduction**: Adaptif öğrenme oranı

## 📝 Sınıflar

Model aşağıdaki 5 sınıfı sınıflandırır:

1. **Hiperakut**: İnmenin ilk saatleri
2. **Akut**: İnmenin ilk günleri
3. **Subakut**: İnmenin ilk haftaları
4. **Kronik**: İnmenin geç dönemi
5. **İnme Yok**: Normal görüntüler

## 🛠️ Geliştirme

### Yeni Özellik Ekleme

1. **Yeni model mimarisi**: `models/model_builder.py` içine ekleyin
2. **Yeni ön işleme**: `utils/image_preprocessing.py` içine ekleyin
3. **Yeni metrikler**: `evaluation/evaluator.py` içine ekleyin

### Veri Seti Kaynağı

Bu projede kullanılan açık veri seti Türkiye Cumhuriyeti Sağlık Bakanlığı Açık Veri Portalı'ndan alınmıştır:
🔗 [Sağlık Bakanlığı Açık Veri - İnme Veri Seti](https://acikveri.saglik.gov.tr/Home/DataSetDetail/1)

### Veri Seti Yapısı

Veri seti yapısı şu şekilde olmalıdır:

```
iskemi/
└── İskemi Veri Seti/
    ├── DICOM/          # DICOM dosyaları
    ├── PNG/            # PNG görüntüleri
    └── OVERLAY/        # Overlay görüntüleri

inmeyok/
└── İnme Yok_kronik süreç_diğer Veri Set_PNG/
    └── *.png          # PNG görüntüleri
```

**Not**: Sınıf etiketleri dosya adlarından veya klasör yapısından otomatik olarak belirlenir. `utils/data_loader.py` içindeki `_determine_stroke_stage()` fonksiyonunu veri setinize göre özelleştirebilirsiniz.

## 📚 Referanslar

- TEKNOFEST 2025 Sağlıkta Yapay Zeka Yarışması
- TensorFlow/Keras Dokümantasyonu
- EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
- Transfer Learning in Medical Imaging

## 📄 Lisans

Bu proje TEKNOFEST 2025 yarışması kapsamında geliştirilmiştir.

## 👥 Katkıda Bulunanlar

Proje ekibi tarafından geliştirilmiştir.

## 📞 İletişim

Sorularınız için lütfen proje ekibi ile iletişime geçin.

---

**Not**: Bu proje, TEKNOFEST 2025 Sağlıkta Yapay Zeka Yarışması gereksinimlerine uygun olarak geliştirilmiştir. Klinik kullanım için ek validasyon ve testler gereklidir.

