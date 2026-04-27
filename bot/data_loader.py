"""
On Numara Veri Yükleyici - Yeni Excel yapısı için
Excel sütunları: no, tarih, no_1 ... no_22
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List

class OnNumaraDataLoader:
    """On Numara çekiliş verilerini yüklemek için sınıf"""
    
    def __init__(self, excel_path: str = "onnumara_2020.xlsx", sheet_name: str = "s1"):
        self.excel_path = Path(excel_path)
        self.sheet_name = sheet_name
        self.df = None
        self.number_columns = [f'no_{i}' for i in range(1, 23)]
        
    def load_data(self) -> pd.DataFrame:
        """Excel dosyasını yükle"""
        if not self.excel_path.exists():
            raise FileNotFoundError(f"Dosya bulunamadı: {self.excel_path}")
        
        self.df = pd.read_excel(self.excel_path, sheet_name=self.sheet_name)
        print(f"✅ Veri yüklendi: {len(self.df)} çekiliş")
        return self.df
    
    def clean_data(self) -> pd.DataFrame:
        """Veri temizleme ve tipleri düzeltme"""
        if self.df is None:
            self.load_data()
        
        # Tarih sütununu datetime'a çevir
        if 'tarih' in self.df.columns:
            self.df['tarih'] = pd.to_datetime(self.df['tarih'], format='%d.%m.%Y', errors='coerce')
        
        # Sayı sütunlarını integer'a çevir
        for col in self.number_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').astype('Int64')
        
        # NaN'ları temizle
        initial = len(self.df)
        self.df = self.df.dropna(subset=['tarih'] + self.number_columns, how='any')
        
        # no sütununu integer yap
        if 'no' in self.df.columns:
            self.df['no'] = pd.to_numeric(self.df['no'], errors='coerce').astype('Int64')
        
        print(f"🧹 Temizlendi: {initial} -> {len(self.df)} satır")
        return self.df
    
    def get_all_numbers(self) -> List[int]:
        """Tüm çekilişlerdeki sayıları döndür"""
        if self.df is None:
            self.load_data()
            self.clean_data()
        return self.df[self.number_columns].values.flatten().tolist()
    
    def get_cekilis(self, index: int) -> List[int]:
        """Belirli bir çekilişin sayılarını döndür"""
        return self.df.iloc[index][self.number_columns].values.tolist()
    
    def get_frequencies(self) -> dict:
        """1-80 arası sayıların frekansı"""
        all_nums = self.get_all_numbers()
        return {num: all_nums.count(num) for num in range(1, 81)}
    
    def split_data(self, train_ratio: float = 0.85) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Eğitim ve test verisine ayır"""
        if self.df is None:
            self.load_data()
            self.clean_data()
        
        split_idx = int(len(self.df) * train_ratio)
        train = self.df.iloc[:split_idx]
        test = self.df.iloc[split_idx:]
        return train, test
    
    def get_summary(self) -> dict:
        """Veri seti özeti"""
        if self.df is None:
            self.load_data()
            self.clean_data()
        
        return {
            'toplam_cekilis': len(self.df),
            'ilk_tarih': self.df['tarih'].min(),
            'son_tarih': self.df['tarih'].max(),
            'toplam_sayi': len(self.df) * 22,
            'benzersiz_sayilar': sorted(self.df[self.number_columns].values.flatten().unique())
        }


if __name__ == "__main__":
    loader = OnNumaraDataLoader()
    loader.load_data()
    loader.clean_data()
    print(loader.get_summary())
