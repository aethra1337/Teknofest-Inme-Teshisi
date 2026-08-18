"""
DICOM görüntü okuma ve dönüştürme modülü
"""

import numpy as np
import pydicom
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DICOMReader:
    """DICOM dosyalarını okuyup PNG formatına dönüştüren sınıf"""
    
    def __init__(
        self,
        window_center: Optional[float] = None,
        window_width: Optional[float] = None,
        rescale_slope: float = 1.0,
        rescale_intercept: float = 0.0,
    ):
        """
        Args:
            window_center: Pencere merkezi (None: otomatik)
            window_width: Pencere genişliği (None: otomatik)
            rescale_slope: Ölçeklendirme eğimi
            rescale_intercept: Ölçeklendirme kesişimi
        """
        self.window_center = window_center
        self.window_width = window_width
        self.rescale_slope = rescale_slope
        self.rescale_intercept = rescale_intercept
    
    def read_dicom(self, dicom_path: Path) -> np.ndarray:
        """
        DICOM dosyasını okur ve numpy array'e dönüştürür
        
        Args:
            dicom_path: DICOM dosya yolu
            
        Returns:
            Görüntü array'i (2D veya 3D)
        """
        try:
            ds = pydicom.dcmread(str(dicom_path))
            
            # Pixel array'i al
            pixel_array = ds.pixel_array.astype(np.float32)
            
            # Rescale slope ve intercept uygula
            if hasattr(ds, 'RescaleSlope'):
                self.rescale_slope = float(ds.RescaleSlope)
            if hasattr(ds, 'RescaleIntercept'):
                self.rescale_intercept = float(ds.RescaleIntercept)
            
            pixel_array = pixel_array * self.rescale_slope + self.rescale_intercept
            
            # Window center ve width ayarları
            if self.window_center is None and hasattr(ds, 'WindowCenter'):
                try:
                    if isinstance(ds.WindowCenter, (list, tuple)):
                        self.window_center = float(ds.WindowCenter[0])
                    elif hasattr(ds.WindowCenter, 'value'):  # MultiValue durumu
                        self.window_center = float(ds.WindowCenter.value[0] if isinstance(ds.WindowCenter.value, (list, tuple)) else ds.WindowCenter.value)
                    else:
                        self.window_center = float(ds.WindowCenter)
                except (ValueError, TypeError, AttributeError):
                    # Varsayılan değerler kullan
                    pass
            
            if self.window_width is None and hasattr(ds, 'WindowWidth'):
                try:
                    if isinstance(ds.WindowWidth, (list, tuple)):
                        self.window_width = float(ds.WindowWidth[0])
                    elif hasattr(ds.WindowWidth, 'value'):  # MultiValue durumu
                        self.window_width = float(ds.WindowWidth.value[0] if isinstance(ds.WindowWidth.value, (list, tuple)) else ds.WindowWidth.value)
                    else:
                        self.window_width = float(ds.WindowWidth)
                except (ValueError, TypeError, AttributeError):
                    # Varsayılan değerler kullan
                    pass
            
            # Window leveling uygula
            if self.window_center is not None and self.window_width is not None:
                pixel_array = self._apply_window_level(
                    pixel_array, self.window_center, self.window_width
                )
            
            return pixel_array
            
        except Exception as e:
            logger.error(f"DICOM okuma hatası ({dicom_path}): {str(e)}")
            raise
    
    def _apply_window_level(
        self, pixel_array: np.ndarray, center: float, width: float
    ) -> np.ndarray:
        """
        Window leveling uygular
        
        Args:
            pixel_array: Pixel array
            center: Pencere merkezi
            width: Pencere genişliği
            
        Returns:
            Window leveling uygulanmış array
        """
        min_val = center - width / 2
        max_val = center + width / 2
        
        pixel_array = np.clip(pixel_array, min_val, max_val)
        pixel_array = (pixel_array - min_val) / (max_val - min_val)
        
        return pixel_array
    
    def dicom_to_png(
        self, dicom_path: Path, output_path: Optional[Path] = None
    ) -> np.ndarray:
        """
        DICOM dosyasını PNG formatına dönüştürür
        
        Args:
            dicom_path: DICOM dosya yolu
            output_path: Çıktı PNG dosya yolu (None: dönüştürme yapılmaz)
            
        Returns:
            Normalize edilmiş görüntü array'i (0-255)
        """
        pixel_array = self.read_dicom(dicom_path)
        
        # Normalize et (0-255)
        if pixel_array.max() > pixel_array.min():
            pixel_array = (
                (pixel_array - pixel_array.min())
                / (pixel_array.max() - pixel_array.min())
                * 255
            )
        else:
            pixel_array = np.zeros_like(pixel_array)
        
        pixel_array = pixel_array.astype(np.uint8)
        
        # PNG olarak kaydet
        if output_path is not None:
            if len(pixel_array.shape) == 2:
                img = Image.fromarray(pixel_array, mode='L')
                img.save(output_path)
            elif len(pixel_array.shape) == 3:
                # 3D görüntü için ilk slice'ı al
                img = Image.fromarray(pixel_array[:, :, 0], mode='L')
                img.save(output_path)
            else:
                raise ValueError(f"Desteklenmeyen görüntü boyutu: {pixel_array.shape}")
        
        return pixel_array
    
    def get_dicom_metadata(self, dicom_path: Path) -> dict:
        """
        DICOM dosyasından metadata bilgilerini çıkarır
        
        Args:
            dicom_path: DICOM dosya yolu
            
        Returns:
            Metadata dictionary
        """
        try:
            ds = pydicom.dcmread(str(dicom_path))
            metadata = {
                "PatientID": getattr(ds, 'PatientID', 'Unknown'),
                "StudyDate": getattr(ds, 'StudyDate', 'Unknown'),
                "Modality": getattr(ds, 'Modality', 'Unknown'),
                "SeriesDescription": getattr(ds, 'SeriesDescription', 'Unknown'),
                "SliceThickness": getattr(ds, 'SliceThickness', None),
                "PixelSpacing": getattr(ds, 'PixelSpacing', None),
            }
            return metadata
        except Exception as e:
            logger.error(f"Metadata okuma hatası ({dicom_path}): {str(e)}")
            return {}

