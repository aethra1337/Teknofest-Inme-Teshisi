"""
Basit Test Scripti - Yeni Başlayanlar İçin
Bu script, modelin çalışıp çalışmadığını test eder (eğitim yapmadan)
"""

import sys
from pathlib import Path

# Proje kök dizinini path'e ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)


def test_imports():
    """Gerekli kütüphanelerin import edilip edilemediğini test eder"""
    logger.info("=" * 80)
    logger.info("KÜTÜPHANE TESTİ")
    logger.info("=" * 80)
    
    try:
        import tensorflow as tf
        logger.info(f"✅ TensorFlow yüklü: {tf.__version__}")
    except ImportError as e:
        logger.error(f"❌ TensorFlow yüklenemedi: {e}")
        return False
    
    try:
        from tensorflow import keras
        logger.info(f"✅ Keras yüklü: {keras.__version__}")
    except ImportError as e:
        logger.error(f"❌ Keras yüklenemedi: {e}")
        return False
    
    try:
        import numpy as np
        logger.info(f"✅ NumPy yüklü: {np.__version__}")
    except ImportError as e:
        logger.error(f"❌ NumPy yüklenemedi: {e}")
        return False
    
    try:
        import pandas as pd
        logger.info(f"✅ Pandas yüklü: {pd.__version__}")
    except ImportError as e:
        logger.error(f"❌ Pandas yüklenemedi: {e}")
        return False
    
    try:
        import cv2
        logger.info(f"✅ OpenCV yüklü: {cv2.__version__}")
    except ImportError as e:
        logger.error(f"❌ OpenCV yüklenemedi: {e}")
        return False
    
    try:
        from PIL import Image
        logger.info("✅ PIL (Pillow) yüklü")
    except ImportError as e:
        logger.error(f"❌ PIL yüklenemedi: {e}")
        return False
    
    try:
        import sklearn
        logger.info(f"✅ Scikit-learn yüklü: {sklearn.__version__}")
    except ImportError as e:
        logger.error(f"❌ Scikit-learn yüklenemedi: {e}")
        return False
    
    return True


def test_model_creation():
    """Model oluşturmayı test eder"""
    logger.info("\n" + "=" * 80)
    logger.info("MODEL OLUŞTURMA TESTİ")
    logger.info("=" * 80)
    
    try:
        from models.model_builder import build_model
        import config
        
        logger.info("Model oluşturuluyor...")
        model = build_model(
            input_shape=config.MODEL_CONFIG["input_shape"],
            num_classes=config.MODEL_CONFIG["num_classes"],
            base_model_name=config.TRANSFER_LEARNING["base_model"],
            dropout_rate=config.MODEL_CONFIG["dropout_rate"],
        )
        
        logger.info(f"✅ Model başarıyla oluşturuldu!")
        logger.info(f"   - Model adı: {model.name}")
        logger.info(f"   - Toplam parametre sayısı: {model.count_params():,}")
        
        # Test input ile tahmin yap
        test_input = np.random.random((1, 224, 224, 3))
        prediction = model.predict(test_input, verbose=0)
        logger.info(f"✅ Test tahmini başarılı!")
        logger.info(f"   - Çıkış şekli: {prediction.shape}")
        logger.info(f"   - Tahmin değerleri: {prediction[0]}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model oluşturma hatası: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_data_loader():
    """Veri yükleyiciyi test eder"""
    logger.info("\n" + "=" * 80)
    logger.info("VERİ YÜKLEYİCİ TESTİ")
    logger.info("=" * 80)
    
    try:
        from utils.data_loader import DataLoader
        import config
        
        logger.info("Veri yükleyici oluşturuluyor...")
        data_loader = DataLoader(
            iskemi_dir=config.ISKEMI_DIR,
            inmeyok_dir=config.INMEYOK_DIR,
            processed_dir=config.PROCESSED_DATA_DIR,
        )
        
        logger.info("✅ Veri yükleyici başarıyla oluşturuldu!")
        
        # Veri yollarını kontrol et
        if config.ISKEMI_DIR.exists():
            logger.info(f"✅ İskemi veri seti bulundu: {config.ISKEMI_DIR}")
        else:
            logger.warning(f"⚠️  İskemi veri seti bulunamadı: {config.ISKEMI_DIR}")
        
        if config.INMEYOK_DIR.exists():
            logger.info(f"✅ İnme yok veri seti bulundu: {config.INMEYOK_DIR}")
        else:
            logger.warning(f"⚠️  İnme yok veri seti bulunamadı: {config.INMEYOK_DIR}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Veri yükleyici hatası: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Ana test fonksiyonu"""
    logger.info("=" * 80)
    logger.info("BASİT TEST - MODEL GELİŞTİRME")
    logger.info("=" * 80)
    logger.info("\nBu script, modelin temel bileşenlerini test eder.")
    logger.info("Eğitim yapmaz, sadece her şeyin çalışıp çalışmadığını kontrol eder.\n")
    
    results = []
    
    # Test 1: Kütüphaneler
    results.append(("Kütüphaneler", test_imports()))
    
    # Test 2: Model oluşturma
    if results[0][1]:  # Eğer kütüphaneler yüklüyse
        results.append(("Model Oluşturma", test_model_creation()))
    else:
        logger.warning("⚠️  Kütüphaneler yüklü olmadığı için model testi atlandı")
        results.append(("Model Oluşturma", False))
    
    # Test 3: Veri yükleyici
    results.append(("Veri Yükleyici", test_data_loader()))
    
    # Sonuçları özetle
    logger.info("\n" + "=" * 80)
    logger.info("TEST SONUÇLARI")
    logger.info("=" * 80)
    
    for test_name, success in results:
        status = "✅ BAŞARILI" if success else "❌ BAŞARISIZ"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        logger.info("\n🎉 Tüm testler başarılı! Eğitime hazırsınız.")
        logger.info("\nSonraki adımlar:")
        logger.info("1. Veri setini kontrol edin: python check_data.py")
        logger.info("2. Eğitimi başlatın: python train.py")
    else:
        logger.error("\n❌ Bazı testler başarısız. Lütfen sorunları giderin.")
        logger.info("\nYardım için:")
        logger.info("- KURULUM_ADIMLARI.md dosyasına bakın")
        logger.info("- BASLANGIC_REHBERI.md dosyasındaki 'Sorun Giderme' bölümüne bakın")
    
    logger.info("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

