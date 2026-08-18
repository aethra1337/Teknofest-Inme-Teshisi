# Loss (Kayıp) Değeri Açıklaması

## Loss Nedir?

**Loss (Kayıp)**, modelin tahminlerinin gerçek değerlerden ne kadar uzak olduğunu gösteren bir metrikdir. Düşük loss = iyi model, yüksek loss = kötü model.

## Loss: 1 Ne Anlama Gelir?

### Categorical Crossentropy için:

Bizim modelimizde **categorical_crossentropy** kullanıyoruz. Bu loss fonksiyonu için:

- **Loss: 0.0 - 0.3**: Çok iyi! Model çok iyi öğreniyor
- **Loss: 0.3 - 0.7**: İyi, model öğreniyor
- **Loss: 0.7 - 1.0**: Orta, model öğrenmeye başlıyor
- **Loss: 1.0 - 2.0**: Kötü, model henüz öğrenmemiş
- **Loss: > 2.0**: Çok kötü, model rastgele tahmin yapıyor gibi

### Loss: 1 Özel Durumu

**Loss: 1** genellikle şu anlama gelir:

1. **İlk Epoch**: Normal! Model henüz öğrenmeye başlamış
2. **Rastgele Tahmin**: Model henüz hiçbir şey öğrenmemiş (2 sınıf için loss ≈ 0.693)
3. **Overfitting**: Model eğitim verisini ezberliyor ama genelleme yapamıyor

### 2 Sınıf İçin Özel:

2 sınıflı bir problemde:
- **Rastgele tahmin**: Loss ≈ 0.693 (log(2))
- **Loss: 1**: Model rastgele tahmin yapıyor gibi (biraz daha kötü)
- **Loss: 0.5**: Model %75 doğrulukta tahmin yapıyor
- **Loss: 0.2**: Model %90+ doğrulukta tahmin yapıyor

## Eğitim Sırasında Loss Değişimi

Normal bir eğitim sürecinde loss şöyle değişir:

```
Epoch 1: loss: 1.0 → 0.8 (öğrenmeye başlıyor)
Epoch 2: loss: 0.8 → 0.5 (iyileşiyor)
Epoch 3: loss: 0.5 → 0.3 (daha da iyileşiyor)
...
```

## Ne Yapmalı?

### Loss: 1 İlk Epoch'ta:
✅ **Normal** - Bekleyin, model öğrenmeye başlayacak

### Loss: 1 Son Epoch'larda:
❌ **Sorun var** - Şunları deneyin:
1. Daha fazla epoch eğitin
2. Learning rate'i ayarlayın
3. Model mimarisini değiştirin
4. Veri augmentation ekleyin

### Loss Azalmıyorsa:
- Learning rate çok yüksek/çok düşük olabilir
- Model yeterince büyük değil
- Veri seti yeterli değil
- Overfitting oluyor olabilir

## Validation Loss

- **Train Loss < Val Loss**: Normal (biraz fark olabilir)
- **Train Loss << Val Loss**: Overfitting! (Model ezberliyor)
- **Train Loss ≈ Val Loss**: İdeal durum

## Örnek Senaryolar

### Senaryo 1: İyi Eğitim
```
Epoch 1: loss: 1.0, val_loss: 1.1
Epoch 2: loss: 0.6, val_loss: 0.7
Epoch 3: loss: 0.4, val_loss: 0.5
```
✅ Model öğreniyor!

### Senaryo 2: Overfitting
```
Epoch 1: loss: 1.0, val_loss: 1.1
Epoch 2: loss: 0.3, val_loss: 0.8
Epoch 3: loss: 0.1, val_loss: 0.9
```
❌ Model ezberliyor, genelleme yapamıyor

### Senaryo 3: Öğrenemiyor
```
Epoch 1: loss: 1.0, val_loss: 1.0
Epoch 2: loss: 1.0, val_loss: 1.0
Epoch 3: loss: 1.0, val_loss: 1.0
```
❌ Model hiç öğrenemiyor, ayarları değiştirin

## Bizim Modelimiz İçin

2 sınıflı problemimizde:
- **İlk epoch'ta loss: 1.0**: Normal, beklenen
- **Son epoch'ta loss: 0.3-0.5**: İyi sonuç
- **Son epoch'ta loss: < 0.3**: Çok iyi sonuç

## İpuçları

1. **Sabırlı olun**: İlk epoch'ta loss yüksek olabilir
2. **İzleyin**: Loss'un azalıp azalmadığına bakın
3. **Karşılaştırın**: Train ve validation loss'u karşılaştırın
4. **Early Stopping**: Loss azalmıyorsa durdurun

---

**Özet**: Loss: 1 ilk epoch'ta normaldir. Önemli olan, loss'un zamanla azalıp azalmadığıdır!

