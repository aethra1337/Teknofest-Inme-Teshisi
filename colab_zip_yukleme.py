"""
Colab'e ZIP Dosyası Yükleme ve Açma Scripti
Bu kodu Colab'de çalıştırın
"""

# ============================================================================
# YÖNTEM 1: Google Drive'dan Yükleme (ÖNERİLEN)
# ============================================================================

from google.colab import drive
import zipfile
import os

# 1. Drive'ı bağlayın
print("🔗 Google Drive bağlanıyor...")
drive.mount('/content/drive')

# 2. ZIP dosyasının yolunu belirtin
# ⚠️ YOLUNUZU DÜZENLEYİN!
zip_path = '/content/drive/MyDrive/Teknofest.zip'

# Alternatif: ZIP dosyasını bulun
if not os.path.exists(zip_path):
    print("\n🔍 ZIP dosyası aranıyor...")
    import glob
    zip_files = glob.glob('/content/drive/MyDrive/**/*.zip', recursive=True)
    if zip_files:
        print("Bulunan ZIP dosyaları:")
        for i, zf in enumerate(zip_files):
            print(f"  {i+1}. {zf}")
        zip_path = zip_files[0]  # İlkini kullan
        print(f"\n✅ Kullanılan: {zip_path}")
    else:
        print("❌ ZIP dosyası bulunamadı!")
        print("💡 Lütfen ZIP dosyasını Google Drive'a yükleyin")

# 3. ZIP'i açın
if os.path.exists(zip_path):
    print(f"\n📦 ZIP açılıyor: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('/content/')
    print("✅ ZIP açıldı!")
    
    # 4. Proje klasörüne geçin
    # ZIP açıldıktan sonra klasör adı genellikle ZIP adıyla aynıdır
    project_name = os.path.splitext(os.path.basename(zip_path))[0]
    project_path = f'/content/{project_name}'
    
    if os.path.exists(project_path):
        os.chdir(project_path)
    else:
        # Alternatif: Teknofest klasörünü bul
        if os.path.exists('/content/Teknofest'):
            os.chdir('/content/Teknofest')
        else:
            print("⚠️ Proje klasörü bulunamadı, mevcut dizinler:")
            !ls -la /content/
    
    print(f"✅ Proje dizini: {os.getcwd()}")
    
    # 5. Dosya yapısını gösterin
    print("\n📁 Proje yapısı:")
    !ls -la
    
else:
    print("❌ ZIP dosyası bulunamadı!")


# ============================================================================
# YÖNTEM 2: Doğrudan Dosya Yükleme (Küçük Dosyalar İçin)
# ============================================================================

# Yukarıdaki yöntem çalışmazsa bu yöntemi kullanın:

"""
from google.colab import files
import zipfile
import os

print("📤 ZIP dosyasını seçin...")
uploaded = files.upload()

if uploaded:
    zip_filename = list(uploaded.keys())[0]
    print(f"✅ Yüklenen: {zip_filename}")
    
    # ZIP'i aç
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall('/content/')
    
    # Proje klasörüne geç
    project_name = os.path.splitext(zip_filename)[0]
    if os.path.exists(f'/content/{project_name}'):
        os.chdir(f'/content/{project_name}')
    elif os.path.exists('/content/Teknofest'):
        os.chdir('/content/Teknofest')
    
    print(f"✅ Proje dizini: {os.getcwd()}")
"""

