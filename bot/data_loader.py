"""
On Numara veri yükleyici modülü
Excel dosyasını okur, temizler ve analize hazır hale getirir
"""

import pandas as pd
import numpy as np
from pathlib import Path

class OnNumaraDataLoader:
    """On Numara çekiliş verilerini yüklemek ve temizlemek için sınıf"""
    
    def __init__(self, excel_path="onnumara_2020.xlsx"):
        """
        Başlatıcı metod
        
        Args:
            excel_path (str): Excel dosyasının yolu (ana dizinde)
        """
        self.excel_path = Path(excel_path)
        self.df = None
        self.number_columns = [f'no-{i}' for i in range(1, 23)]
        
    def load_data(self):
        """Excel dosyasını yükle ve temel kontrolleri yap"""
        if not self.excel_path.exists():
            raise FileNotFoundError(f"Dosya bulunamadı: {self.excel_path}")
        
        self.df = pd.read_excel(self.excel_path)
        print(f"✅ Veri yüklendi: {len(self.df)} çekiliş, {len(self.df.columns)} sütun")
        return self.df
    
    def clean_data(self):
        """Sütun isimlerini temizle ve veri tiplerini düzelt"""
        # Sütun isimlerini küçült ve boşlukları temizle
        self.df.columns = self.df.columns.str.strip().str.lower()
        
        # Tarih sütununu datetime'a çevir
        if 'tarih' in self.df.columns:
            self.df['tarih'] = pd.to_datetime(self.df['tarih'], format='%d.%m.%Y', errors='coerce')
        
        # Sayı sütunlarını integer'a çevir
        for col in self.number_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Geçersiz satırları temizle (tarihi veya sayıları NaN olanlar)
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=['tarih'])
        self.df = self.df.dropna(subset=self.number_columns, how='any')
        
        print(f"🧹 Veri temizlendi: {initial_count} -> {len(self.df)} satır")
        return self.df
    
    def get_all_numbers(self):
        """Tüm çekilişlerdeki sayıları tek bir liste olarak döndür"""
        if self.df is None:
            self.load_data()
        return self.df[self.number_columns].values.flatten().tolist()
    
    def get_cekilis_by_index(self, index):
        """Belirli bir çekilişin sayılarını döndür"""
        if self.df is None:
            self.load_data()
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
            'ilk_tarih': self.df['tarih'].min(),
            'son_tarih': self.df['tarih'].max(),
            'toplam_sayi_gozlemi': len(self.df) * 22,
            'benzersiz_sayilar': sorted(self.df[self.number_columns].values.flatten().unique())
        }
        return stats


# Kullanım örneği (doğrudan çalıştırma için)
if __name__ == "__main__":
    loader = OnNumaraDataLoader()
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
