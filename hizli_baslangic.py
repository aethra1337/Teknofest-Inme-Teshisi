"""
Hızlı Başlangıç Scripti
Bu script, yeni başlayanlar için basit bir test çalıştırır.
"""

import sys
from pathlib import Path

# Proje kök dizinini path'e ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import logging
import config

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)


def check_environment():
    """Ortamı kontrol eder"""
    logger.info("=" * 80)
    logger.info("ORTAM KONTROLÜ")
    logger.info("=" * 80)
    
    # Python versiyonu
    import sys
    logger.info(f"Python versiyonu: {sys.version}")
    
    # Gerekli kütüphaneleri kontrol et
    required_packages = [
        'tensorflow',
        'keras',
        'numpy',
        'pandas',
        'opencv-python',
        'PIL',
        'sklearn',
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'opencv-python':
                __import__('cv2')
            elif package == 'PIL':
                __import__('PIL')
            else:
                __import__(package)
            logger.info(f"✅ {package} yüklü")
        except ImportError:
            logger.warning(f"❌ {package} yüklü değil")
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"\nEksik paketler: {', '.join(missing_packages)}")
        logger.error("Lütfen şu komutu çalıştırın: pip install -r requirements.txt")
        return False
    
    # GPU kontrolü
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"✅ {len(gpus)} GPU bulundu")
            for gpu in gpus:
                logger.info(f"   - {gpu}")
        else:
            logger.warning("⚠️  GPU bulunamadı, CPU kullanılacak (daha yavaş)")
    except Exception as e:
        logger.warning(f"⚠️  GPU kontrolü başarısız: {e}")
    
    # Veri seti yollarını kontrol et
    logger.info("\n" + "=" * 80)
    logger.info("VERİ SETİ KONTROLÜ")
    logger.info("=" * 80)
    
    iskemi_dir = config.ISKEMI_DIR
    inmeyok_dir = config.INMEYOK_DIR
    
    if iskemi_dir.exists():
        png_count = len(list((iskemi_dir / "PNG").glob("*.png"))) if (iskemi_dir / "PNG").exists() else 0
        dicom_count = len(list((iskemi_dir / "DICOM").glob("*.dcm"))) if (iskemi_dir / "DICOM").exists() else 0
        logger.info(f"✅ İskemi veri seti bulundu: {iskemi_dir}")
        logger.info(f"   - PNG dosyaları: {png_count}")
        logger.info(f"   - DICOM dosyaları: {dicom_count}")
    else:
        logger.error(f"❌ İskemi veri seti bulunamadı: {iskemi_dir}")
        return False
    
    if inmeyok_dir.exists():
        png_count = len(list(inmeyok_dir.glob("*.png")))
        logger.info(f"✅ İnme yok veri seti bulundu: {inmeyok_dir}")
        logger.info(f"   - PNG dosyaları: {png_count}")
    else:
        logger.error(f"❌ İnme yok veri seti bulundu: {inmeyok_dir}")
        return False
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ ORTAM HAZIR!")
    logger.info("=" * 80)
    logger.info("\nSonraki adımlar:")
    logger.info("1. Veri setini kontrol edin: python check_data.py")
    logger.info("2. Eğitimi başlatın: python train.py")
    logger.info("\nDetaylı bilgi için BASLANGIC_REHBERI.md dosyasına bakın.")
    
    return True


def main():
    """Ana fonksiyon"""
    success = check_environment()
    
    if not success:
        logger.error("\n❌ Ortam kontrolü başarısız. Lütfen sorunları giderin.")
        sys.exit(1)
    
    logger.info("\n🎉 Her şey hazır! Eğitime başlayabilirsiniz.")


if __name__ == "__main__":
    main()

