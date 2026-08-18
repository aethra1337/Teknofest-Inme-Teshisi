# Önemli Notlar

## Mevcut Durum

1. **Veri Seti**: Şu anda sadece 2 sınıf var:
   - Akut (2260 görüntü)
   - İnme Yok (4427 görüntü)

2. **Model Beklentisi**: Model 5 sınıf bekliyor:
   - Hiperakut
   - Akut
   - Subakut
   - Kronik
   - İnme Yok

## Çözüm Seçenekleri

### Seçenek 1: Model'i 2 Sınıfa Göre Ayarlamak
- `config.py` içinde `num_classes: 2` yapın
- Daha basit ve hızlı
- TEKNOFEST gereksinimlerine uymayabilir

### Seçenek 2: Veri Setindeki Sınıfları Daha İyi Belirlemek
- DICOM metadata'sından veya dosya adlarından sınıfları belirleyin
- Veri seti yapısını inceleyin
- Daha fazla çalışma gerektirir

## Şu Anda Yapılması Gerekenler

1. ✅ Görüntü kopyalama fonksiyonu eklendi
2. ⚠️ Sınıf sayısını veri setine göre ayarlayın
3. ⚠️ Veri setindeki sınıfları daha iyi belirleyin

## Hızlı Test İçin

`config.py` dosyasında:
```python
MODEL_CONFIG = {
    "num_classes": 2,  # 5 yerine 2
    ...
}
```

Bu şekilde hızlı bir test yapabilirsiniz.

