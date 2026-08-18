# GPU Kurulum Scripti - TensorFlow GPU Versiyonu
# RTX 3060 için

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "GPU KURULUM SCRIPTI" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Adım 1: Mevcut TensorFlow'u kaldır
Write-Host "Adım 1: Mevcut TensorFlow kaldırılıyor..." -ForegroundColor Yellow
pip uninstall tensorflow tensorflow-cpu -y

Write-Host ""
Write-Host "Adım 2: TensorFlow GPU versiyonu kuruluyor..." -ForegroundColor Yellow
Write-Host "Bu işlem birkaç dakika sürebilir..." -ForegroundColor Gray

# TensorFlow GPU versiyonu
pip install tensorflow[and-cuda]

Write-Host ""
Write-Host "Adım 3: Kurulum kontrol ediliyor..." -ForegroundColor Yellow
python gpu_kontrol.py

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "KURULUM TAMAMLANDI!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Eğer GPU bulundu mesajı görürseniz, kurulum başarılı!" -ForegroundColor Green
Write-Host "Eğitim için: python train_akut_20epoch_local.py" -ForegroundColor Cyan

