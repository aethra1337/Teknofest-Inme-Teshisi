"""
Görüntü ön işleme modülü
"""

import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Görüntü ön işleme sınıfı"""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalization: str = "imagenet",
    ):
        """
        Args:
            target_size: Hedef görüntü boyutu (height, width)
            normalization: Normalizasyon tipi ('imagenet', 'custom', None)
        """
        self.target_size = target_size
        self.normalization = normalization
        
        # ImageNet normalizasyon değerleri
        self.imagenet_mean = np.array([0.485, 0.456, 0.406])
        self.imagenet_std = np.array([0.229, 0.224, 0.225])
    
    def load_image(self, image_path: Path) -> np.ndarray:
        """
        Görüntüyü yükler
        
        Args:
            image_path: Görüntü dosya yolu
            
        Returns:
            Görüntü array'i (RGB, uint8)
        """
        try:
            # OpenCV ile yükle (BGR formatında)
            img = cv2.imread(str(image_path))
            if img is None:
                # PIL ile dene
                img = np.array(Image.open(image_path))
                if len(img.shape) == 2:
                    # Grayscale'i RGB'ye çevir
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif len(img.shape) == 3 and img.shape[2] == 4:
                    # RGBA'yı RGB'ye çevir
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            else:
                # BGR'yi RGB'ye çevir
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            return img
            
        except Exception as e:
            logger.error(f"Görüntü yükleme hatası ({image_path}): {str(e)}")
            raise
    
    def resize_image(
        self, image: np.ndarray, interpolation: int = cv2.INTER_LINEAR
    ) -> np.ndarray:
        """
        Görüntüyü yeniden boyutlandırır
        
        Args:
            image: Görüntü array'i
            interpolation: Interpolasyon metodu
            
        Returns:
            Yeniden boyutlandırılmış görüntü
        """
        return cv2.resize(image, (self.target_size[1], self.target_size[0]), interpolation=interpolation)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Görüntüyü normalize eder
        
        Args:
            image: Görüntü array'i (0-255, uint8)
            
        Returns:
            Normalize edilmiş görüntü (float32)
        """
        # 0-1 aralığına normalize et
        image = image.astype(np.float32) / 255.0
        
        if self.normalization == "imagenet":
            # ImageNet normalizasyonu
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = (image - self.imagenet_mean) / self.imagenet_std
            else:
                # Grayscale için ortalama ve std kullan
                mean = np.mean(image)
                std = np.std(image)
                if std > 0:
                    image = (image - mean) / std
        elif self.normalization == "custom":
            # Özel normalizasyon (z-score)
            mean = np.mean(image)
            std = np.std(image)
            if std > 0:
                image = (image - mean) / std
        
        return image
    
    def preprocess_image(
        self, image_path: Path, normalize: bool = True
    ) -> np.ndarray:
        """
        Görüntüyü tam ön işleme pipeline'ından geçirir
        
        Args:
            image_path: Görüntü dosya yolu
            normalize: Normalizasyon uygulanacak mı
            
        Returns:
            Ön işlenmiş görüntü array'i
        """
        # Görüntüyü yükle
        image = self.load_image(image_path)
        
        # Yeniden boyutlandır
        image = self.resize_image(image)
        
        # Normalize et
        if normalize:
            image = self.normalize_image(image)
        
        return image
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        CLAHE (Contrast Limited Adaptive Histogram Equalization) uygular
        
        Args:
            image: Görüntü array'i (grayscale veya RGB)
            
        Returns:
            CLAHE uygulanmış görüntü
        """
        if len(image.shape) == 2:
            # Grayscale
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
        else:
            # RGB
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    def apply_gaussian_blur(
        self, image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5)
    ) -> np.ndarray:
        """
        Gaussian blur uygular
        
        Args:
            image: Görüntü array'i
            kernel_size: Kernel boyutu
            
        Returns:
            Blur uygulanmış görüntü
        """
        return cv2.GaussianBlur(image, kernel_size, 0)
    
    def apply_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """
        Histogram eşitleme uygular
        
        Args:
            image: Görüntü array'i (grayscale)
            
        Returns:
            Histogram eşitlenmiş görüntü
        """
        if len(image.shape) == 2:
            return cv2.equalizeHist(image)
        else:
            # RGB için her kanalı ayrı ayrı eşitle
            yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

