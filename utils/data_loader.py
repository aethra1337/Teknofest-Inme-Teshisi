"""
Veri yükleme ve organizasyon modülü
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging
from sklearn.model_selection import train_test_split
import shutil

from .dicom_reader import DICOMReader
from .image_preprocessing import ImagePreprocessor

logger = logging.getLogger(__name__)


class DataLoader:
    """Veri seti yükleme ve organizasyon sınıfı"""
    
    def __init__(
        self,
        iskemi_dir: Path,
        inmeyok_dir: Path,
        processed_dir: Path,
        dicom_reader: Optional[DICOMReader] = None,
        image_preprocessor: Optional[ImagePreprocessor] = None,
    ):
        """
        Args:
            iskemi_dir: İskemi veri seti dizini
            inmeyok_dir: İnme yok veri seti dizini
            processed_dir: İşlenmiş veri dizini
            dicom_reader: DICOM okuyucu (None: otomatik oluşturulur)
            image_preprocessor: Görüntü ön işleyici (None: otomatik oluşturulur)
        """
        self.iskemi_dir = Path(iskemi_dir)
        self.inmeyok_dir = Path(inmeyok_dir)
        self.processed_dir = Path(processed_dir)
        
        self.dicom_reader = dicom_reader or DICOMReader()
        self.image_preprocessor = image_preprocessor or ImagePreprocessor()
        
        # Sınıf eşlemeleri
        self.class_mapping = {
            "Hiperakut": 0,
            "Akut": 1,
            "Subakut": 2,
            "Kronik": 3,
            "İnme Yok": 4,
        }
        self.reverse_class_mapping = {v: k for k, v in self.class_mapping.items()}
    
    def collect_image_paths(self) -> pd.DataFrame:
        """
        Tüm görüntü yollarını toplar ve DataFrame'e dönüştürür
        
        Returns:
            Görüntü yolları ve etiketlerini içeren DataFrame
        """
        image_paths = []
        labels = []
        sources = []
        
        # İskemi veri seti (PNG klasörü) - SADECE PNG KULLAN
        iskemi_png_dir = self.iskemi_dir / "PNG"
        if iskemi_png_dir.exists():
            png_files = list(iskemi_png_dir.glob("*.png"))
            for png_file in png_files:
                image_paths.append(png_file)
                # Dosya adından veya klasör yapısından sınıf belirleme
                # Bu kısım veri seti yapısına göre özelleştirilebilir
                label = self._determine_stroke_stage(png_file)
                labels.append(label)
                sources.append("iskemi_png")
        
        # DICOM klasörü atlanıyor - sadece PNG kullanılıyor
        
        # İnme yok veri seti
        if self.inmeyok_dir.exists():
            inmeyok_files = list(self.inmeyok_dir.glob("*.png"))
            for img_file in inmeyok_files:
                image_paths.append(img_file)
                labels.append("İnme Yok")
                sources.append("inmeyok")
        
        # DataFrame oluştur
        df = pd.DataFrame({
            "image_path": image_paths,
            "label": labels,
            "source": sources,
        })
        
        logger.info(f"Toplam {len(df)} görüntü bulundu")
        logger.info(f"Sınıf dağılımı:\n{df['label'].value_counts()}")
        
        return df
    
    def _determine_stroke_stage(self, file_path: Path) -> str:
        """
        Dosya yolundan veya metadata'dan inme evresini belirler
        
        Not: Bu fonksiyon veri seti yapısına göre özelleştirilmelidir
        
        Args:
            file_path: Görüntü dosya yolu
            
        Returns:
            İnme evresi etiketi
        """
        # Dosya adı veya klasör yapısından evre belirleme
        # Örnek: dosya adında "hiperakut", "akut", "subakut", "kronik" kelimeleri
        file_name_lower = file_path.name.lower()
        parent_dir_lower = str(file_path.parent).lower()
        
        if "hiperakut" in file_name_lower or "hyperacute" in file_name_lower:
            return "Hiperakut"
        elif "akut" in file_name_lower or "acute" in file_name_lower:
            return "Akut"
        elif "subakut" in file_name_lower or "subacute" in file_name_lower:
            return "Subakut"
        elif "kronik" in file_name_lower or "chronic" in file_name_lower:
            return "Kronik"
        else:
            # Varsayılan olarak "Akut" döndür (veri seti yapısına göre değiştirilebilir)
            logger.warning(f"Evre belirlenemedi, varsayılan 'Akut' kullanılıyor: {file_path}")
            return "Akut"
    
    def split_data(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
        stratify: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Veri setini train/val/test olarak böler
        
        Args:
            df: Görüntü DataFrame'i
            train_ratio: Eğitim seti oranı
            val_ratio: Validasyon seti oranı
            test_ratio: Test seti oranı
            random_seed: Rastgele tohum
            stratify: Stratified split kullanılacak mı
            
        Returns:
            (train_df, val_df, test_df)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Oranların toplamı 1.0 olmalı"
        
        # İlk olarak train ve geçici (val+test) setlerine ayır
        stratify_col = df["label"] if stratify else None
        train_df, temp_df = train_test_split(
            df,
            test_size=(1 - train_ratio),
            random_state=random_seed,
            stratify=stratify_col,
        )
        
        # Geçici seti val ve test'e ayır
        val_size = val_ratio / (val_ratio + test_ratio)
        stratify_col = temp_df["label"] if stratify else None
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_size),
            random_state=random_seed,
            stratify=stratify_col,
        )
        
        logger.info(f"Train seti: {len(train_df)} görüntü")
        logger.info(f"Validation seti: {len(val_df)} görüntü")
        logger.info(f"Test seti: {len(test_df)} görüntü")
        
        return train_df, val_df, test_df
    
    def prepare_processed_data(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: Path,
        copy_images: bool = True,
    ):
        """
        İşlenmiş verileri organize eder ve kaydeder
        
        Args:
            train_df: Eğitim DataFrame'i
            val_df: Validasyon DataFrame'i
            test_df: Test DataFrame'i
            output_dir: Çıktı dizini
            copy_images: Görüntüleri kopyala (True) veya sadece metadata kaydet (False)
        """
        output_dir = Path(output_dir)
        
        for split_name, split_df in [
            ("train", train_df),
            ("val", val_df),
            ("test", test_df),
        ]:
            split_dir = output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            # Sadece mevcut sınıflar için klasör oluştur
            unique_labels = split_df['label'].unique()
            for class_name in unique_labels:
                class_dir = split_dir / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
            
            # Görüntüleri kopyala
            if copy_images:
                logger.info(f"{split_name} seti için görüntüler kopyalanıyor...")
                copied_count = 0
                for idx, row in split_df.iterrows():
                    src_path = Path(row['image_path'])
                    label = row['label']
                    dst_dir = split_dir / label
                    
                    # Dosya adını koru
                    dst_path = dst_dir / src_path.name
                    
                    try:
                        # Sadece PNG dosyalarını kopyala
                        if src_path.suffix.lower() == '.png':
                            import shutil
                            shutil.copy2(src_path, dst_path)
                            copied_count += 1
                        else:
                            # DICOM dosyaları atlanıyor
                            logger.debug(f"DICOM dosyası atlandı (sadece PNG kullanılıyor): {src_path}")
                    except Exception as e:
                        logger.warning(f"Görüntü kopyalanamadı ({src_path}): {e}")
                
                logger.info(f"{split_name} seti için {copied_count} görüntü kopyalandı")
            
            # DataFrame'i kaydet
            split_df.to_csv(
                split_dir / f"{split_name}_metadata.csv",
                index=False,
                encoding='utf-8-sig'
            )
            
            logger.info(f"{split_name} seti hazırlandı: {split_dir}")
    
    def get_class_weights(self, df: pd.DataFrame) -> Dict[int, float]:
        """
        Sınıf ağırlıklarını hesaplar (imbalanced data için)
        
        Args:
            df: Görüntü DataFrame'i
            
        Returns:
            Sınıf ağırlıkları dictionary'si
        """
        class_counts = df["label"].value_counts()
        total_samples = len(df)
        num_classes = len(class_counts)
        
        class_weights = {}
        for class_name, count in class_counts.items():
            class_idx = self.class_mapping[class_name]
            # Inverse frequency weighting
            weight = total_samples / (num_classes * count)
            class_weights[class_idx] = weight
        
        logger.info(f"Sınıf ağırlıkları: {class_weights}")
        return class_weights

