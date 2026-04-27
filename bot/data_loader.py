"""
On Numara veri yükleyici modülü
Excel dosyasını okur, temizler ve analize hazır hale getirir
"""

import pandas as pd
import numpy as np
from pathlib import Path

class OnNumaraDataLoader:
    """On Numara çekiliş verilerini yüklemek ve temizlemek için sınıf"""
    
    def __init__(self, excel_path="onnumara_2020.xlsx", sheet_name="Sayfa7"):
        """
        Args:
            excel_path (str): Excel dosyasının yolu
            sheet_name (str): Sayfa adı (Sayfa7 veya Sayfa8)
        """
        self.excel_path = Path(excel_path)
        self.sheet_name = sheet_name
        self.df = None
        self.number_columns = [f'no-{i}' for i in range(1, 23)]
        
    def load_data(self):
        """Excel dosyasını yükle"""
        if not self.excel_path.exists():
            raise FileNotFoundError(f"Dosya bulunamadı: {self.excel_path}")
        
        # Sayfa7'de ilk 2 satır boş olabilir, onları atla
        if self.sheet_name == "Sayfa7":
            self.df = pd.read_excel(self.excel_path, sheet_name=self.sheet_name, header=3)
        else:
            self.df = pd.read_excel(self.excel_path, sheet_name=self.sheet_name)
        
        print(f"✅ Veri yüklendi: {len(self.df)} satır, {len(self.df.columns)} sütun")
        return self.df
    
    def clean_data(self):
        """Sütun isimlerini temizle ve veri tiplerini düzelt"""
        # İlk sütunu (A sütunu) index yap
        if self.df.columns[0] == self.df.columns[0]:  # Unnamed gibi bir şey olabilir
            pass
            
        # Sütun isimlerini küçült ve boşlukları temizle
        self.df.columns = self.df.columns.str.strip().str.lower()
        
        # İlk sütun genelde boş veya gereksiz, onu silelim
        first_col = self.df.columns[0]
        if 'unnamed' in first_col or first_col == '' or first_col == 'nan':
            self.df = self.df.drop(columns=[first_col])
        
        # Tarih sütununu bul (tarih, date veya çekiliş içinde)
        date_col = None
        for col in self.df.columns:
            if 'tarih' in col or 'date' in col or 'çekiliş' in col:
                date_col = col
                break
        
        if date_col:
            # Tarihi parçala (örn: "1.Çekiliş[1] 03/08/2020")
            self.df['tarih_raw'] = self.df[date_col].astype(str)
            # Boşluktan sonraki kısmı al (tarih)
            self.df['tarih_str'] = self.df['tarih_raw'].str.split().str[-1]
            self.df['tarih'] = pd.to_datetime(self.df['tarih_str'], format='%d/%m/%Y', errors='coerce')
            self.df = self.df.drop(columns=[date_col, 'tarih_raw', 'tarih_str'])
        
        # Çekiliş numarasını ayıkla (köşeli parantez içindeki sayı)
        if 'no' not in self.df.columns:
            # İlk sütun artık tarihi içermiyor, onu no yapalım
            if len(self.df.columns) > 0:
                self.df['no'] = range(1, len(self.df) + 1)
        
        # Sayı sütunlarını integer'a çevir
        for col in self.number_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Geçersiz satırları temizle
        initial_count = len(self.df)
        if 'tarih' in self.df.columns:
            self.df = self.df.dropna(subset=['tarih'])
        self.df = self.df.dropna(subset=self.number_columns, how='any')
        
        print(f"🧹 Veri temizlendi: {initial_count} -> {len(self.df)} satır")
        return self.df
    
    def get_all_numbers(self):
        """Tüm çekilişlerdeki sayıları tek bir liste olarak döndür"""
        if self.df is None:
            self.load_data()
            self.clean_data()
        return self.df[self.number_columns].values.flatten().tolist()
    
    def get_cekilis_by_index(self, index):
        """Belirli bir çekilişin sayılarını döndür"""
        if self.df is None:
            self.load_data()
            self.clean_data()
        return self.df.iloc[index][self.number_columns].values.tolist()
    
    def get_number_frequencies(self):
        """Her sayının (1-80) çıkma sıklığını hesapla"""
        all_numbers = self.get_all_numbers()
        frequencies = {}
        for num in range(1, 81):
            frequencies[num] = all_numbers.count(num)
        return frequencies
    
    def get_summary_stats(self):
        """Veri seti hakkında özet istatistik"""
        if self.df is None:
            self.load_data()
            self.clean_data()
        
        stats = {
            'toplam_cekilis': len(self.df),
            'ilk_tarih': self.df['tarih'].min() if 'tarih' in self.df.columns else None,
            'son_tarih': self.df['tarih'].max() if 'tarih' in self.df.columns else None,
            'toplam_sayi_gozlemi': len(self.df) * 22,
            'benzersiz_sayilar': sorted(self.df[self.number_columns].values.flatten().unique())
        }
        return stats


if __name__ == "__main__":
    loader = OnNumaraDataLoader(sheet_name="Sayfa7")
    df = loader.load_data()
    df = loader.clean_data()
    
    print("\n📊 Veri seti özeti:")
    stats = loader.get_summary_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print(f"\n📈 En sık çıkan 10 sayı:")
    freqs = loader.get_number_frequencies()
    top_10 = sorted(freqs.items(), key=lambda x: x[1], reverse=True)[:10]
    for num, count in top_10:
        print(f"   Sayı {num}: {count} kez")
