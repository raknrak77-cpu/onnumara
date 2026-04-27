#!/usr/bin/env python3
"""
ON NUMARA HİBRİT TAHMİN BOTU
- 22'lik botun 40-80 arası en güçlü 6 sayısını alır
- Geri kalan 10 sayıyı rastgele seçer
- Toplam 16 sayı önerir
- Backtest ile optimal stratejiyi bulur
"""

import pandas as pd
import numpy as np
from collections import Counter
import json
import os
import random

# ============================================================
# VERİ YÜKLEYİCİ (SADECE GEÇMİŞ VERİLERİ KULLANIR - HİLE YOK!)
# ============================================================

class DataLoader:
    def __init__(self, excel_path="onnumara_2020.xlsx", sheet_name="s1"):
        self.excel_path = excel_path
        self.sheet_name = sheet_name
        self.df = None
        self.number_columns = [f'no_{i}' for i in range(1, 23)]
        
    def load(self):
        self.df = pd.read_excel(self.excel_path, sheet_name=self.sheet_name, header=0)
        
        if 'tarih' in self.df.columns:
            self.df['tarih'] = pd.to_datetime(self.df['tarih'], format='%d.%m.%Y', errors='coerce')
        
        for col in self.number_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        self.df = self.df.dropna(subset=['tarih'] + self.number_columns, how='any')
        self.df = self.df.reset_index(drop=True)
        return self.df
    
    def get_numbers(self, row):
        nums = []
        for col in self.number_columns:
            if col in row.index and pd.notna(row[col]):
                try:
                    nums.append(int(row[col]))
                except:
                    pass
        return nums


# ============================================================
# 22'LİK MODEL (SADECE 40-80 ARASINI ALIYORUZ)
# ============================================================

class Model22:
    def __init__(self, df, get_numbers_func):
        self.df = df
        self.get_numbers = get_numbers_func
        
    def get_top_40_80_numbers(self, n=6):
        """22'lik modelin 40-80 arasındaki en güçlü n sayısını döndür"""
        
        # Tüm zamanların en sık çıkan sayıları
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        
        counter = Counter(all_nums)
        
        # Sadece 40-80 arasındakileri filtrele
        filtered = [(num, count) for num, count in counter.items() if 40 <= num <= 80]
        filtered.sort(key=lambda x: x[1], reverse=True)
        
        # En sık çıkan n sayıyı al
        result = [num for num, _ in filtered[:n]]
        
        # Eğer yeterli yoksa, 40-80 arasında rastgele tamamla
        if len(result) < n:
            all_40_80 = list(range(40, 81))
            remaining = [num for num in all_40_80 if num not in result]
            random.shuffle(remaining)
            result.extend(remaining[:n - len(result)])
        
        return result


# ============================================================
# HİBRİT TAHMİN BOTU
# ============================================================

class HybridPredictor:
    def __init__(self, excel_path="onnumara_2020.xlsx", sheet_name="s1"):
        self.excel_path = excel_path
        self.sheet_name = sheet_name
        self.df = None
        self.model22 = None
        
    def load_data(self):
        loader = DataLoader(self.excel_path, self.sheet_name)
        self.df = loader.load()
        self.get_numbers = loader.get_numbers
        self.model22 = Model22(self.df, self.get_numbers)
        
        print(f"✅ Veri yüklendi: {len(self.df)} çekiliş")
        print(f"📅 Aralık: {self.df['tarih'].min().strftime('%d.%m.%Y')} - {self.df['tarih'].max().strftime('%d.%m.%Y')}")
        return self.df
    
    def get_hybrid_prediction(self, bot_count=6, random_count=10):
        """
        Hibrit tahmin üretir:
        - bot_count: 22'lik modelden 40-80 arası kaç sayı
        - random_count: Rastgele kaç sayı (1-80 arası)
        - Toplam: bot_count + random_count sayı önerir
        """
        
        # 1. 22'lik modelden en güçlü 40-80 arası sayıları al
        bot_numbers = self.model22.get_top_40_80_numbers(bot_count)
        
        # 2. Rastgele sayılar üret (1-80 arası, bot_numbers ile çakışmayacak)
        all_numbers = set(range(1, 81))
        used = set(bot_numbers)
        available = list(all_numbers - used)
        
        random.shuffle(available)
        random_numbers = available[:random_count]
        
        # 3. Birleştir ve karıştır
        final_numbers = bot_numbers + random_numbers
        random.shuffle(final_numbers)
        
        return {
            'bot_numbers': bot_numbers,
            'random_numbers': random_numbers,
            'final_numbers': final_numbers,
            'total': len(final_numbers)
        }
    
    def run_backtest(self, test_size=50, bot_count=6, random_count=10):
        """
        Geriye dönük test - son test_size çekiliş üzerinde
        HİLE YOK - SADECE GEÇMİŞ VERİLER KULLANILIR
        """
        total = len(self.df)
        train_size = total - test_size
        
        if train_size <= 0:
            return None
        
        scores = []
        
        for i in range(test_size):
            train_end = train_size + i
            if train_end >= total:
                break
            
            # Eğitim verisi (sonuçları bilmiyoruz - sadece geçmiş)
            train_df = self.df.iloc[:train_end]
            
            # Test verisi (gerçek sonuç - backtest için kullanıyoruz)
            test_row = self.df.iloc[train_end]
            actual_numbers = set(self.get_numbers(test_row))
            
            # Geçici model ile tahmin yap (geçmiş verileri kullanarak)
            temp_model = Model22(train_df, self.get_numbers)
            temp_bot_numbers = temp_model.get_top_40_80_numbers(bot_count)
            
            # Rastgele sayılar (her backtest farklı olacak - ortalamayı alıyoruz)
            all_nums = set(range(1, 81))
            used = set(temp_bot_numbers)
            available = list(all_nums - used)
            random.shuffle(available)
            temp_random_numbers = available[:random_count]
            
            temp_final = set(temp_bot_numbers + temp_random_numbers)
            
            # Kaç doğru bilindi?
            correct = len(temp_final & actual_numbers)
            scores.append(correct)
        
        avg_score = sum(scores) / len(scores) if scores else 0
        return {
            'avg_correct': avg_score,
            'total_predictions': bot_count + random_count,
            'bot_count': bot_count,
            'random_count': random_count,
            'test_size': test_size
        }
    
    def optimize(self, test_size=50):
        """En iyi bot_count ve random_count kombinasyonunu bul"""
        print("\n🔍 OPTİMİZASYON YAPILIYOR (HİLE YOK - GEÇMİŞ VERİLERLE)")
        print("-" * 50)
        
        best_score = 0
        best_config = {'bot_count': 6, 'random_count': 10}
        
        # Farklı kombinasyonları dene
        for bot_count in [4, 5, 6, 7, 8]:
            for random_count in [8, 9, 10, 11, 12]:
                total = bot_count + random_count
                if total < 14 or total > 18:
                    continue
                
                result = self.run_backtest(test_size, bot_count, random_count)
                if result:
                    score = result['avg_correct']
                    print(f"   bot={bot_count}, random={random_count}, total={total} -> {score:.2f}/22 doğru")
                    
                    if score > best_score:
                        best_score = score
                        best_config = {'bot_count': bot_count, 'random_count': random_count, 'score': score}
        
        print(f"\n✅ EN İYİ KONFİGÜRASYON: bot={best_config['bot_count']}, random={best_config['random_count']}")
        print(f"   Ortalama {best_score:.2f}/22 doğru (son {test_size} çekilişte)")
        
        return best_config
    
    def print_report(self, config=None):
        """Rapor oluştur"""
        if config is None:
            config = {'bot_count': 6, 'random_count': 10}
        
        # Hibrit tahmin üret
        prediction = self.get_hybrid_prediction(config['bot_count'], config['random_count'])
        
        print("\n" + "=" * 60)
        print("🎯 ON NUMARA HİBRİT TAHMİN BOTU")
        print("   22'lik model (40-80) + Rastgele sayılar")
        print("=" * 60)
        print(f"\n📅 Son Çekiliş: {self.df['tarih'].max().strftime('%d.%m.%Y')}")
        print(f"📊 Toplam Analiz: {len(self.df)} çekiliş")
        
        print("\n" + "-" * 60)
        print("📋 TAHMİN DETAYLARI")
        print("-" * 60)
        
        print(f"\n🤖 22'lik Modelden (40-80 arası):")
        print(f"   {prediction['bot_numbers']}")
        
        print(f"\n🎲 Rastgele Seçilenler (1-80 arası):")
        print(f"   {prediction['random_numbers'][:5]}... (toplam {len(prediction['random_numbers'])})")
        
        print("\n" + "-" * 60)
        print("🏆 ÖNERİLEN 16 SAYI (KARIŞTIRILMIŞ)")
        print("-" * 60)
        
        final = prediction['final_numbers']
        for i in range(0, len(final), 4):
            group = final[i:i+4]
            print(f"  {i+1:2d}-{i+4:2d}. {' '.join(f'{n:3d}' for n in group)}")
        
        print("\n" + "-" * 60)
        print("📊 STRATEJİ ÖZETİ")
        print("-" * 60)
        print(f"   🤖 Bot güvenli sayılar: {config['bot_count']} adet (40-80 arası)")
        print(f"   🎲 Rastgele sayılar: {config['random_count']} adet (1-80 arası)")
        print(f"   📈 Toplam önerilen: {config['bot_count'] + config['random_count']} sayı")
        
        print("\n" + "-" * 60)
        print("⚠️ NOT: Bu tahminler istatistiksel analizdir.")
        print("   Kesin sonuç garantisi yoktur. Eğlence amaçlıdır.")
        print("=" * 60)
        
        return prediction


# ============================================================
# ANA ÇALIŞTIRMA
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("🚀 ON NUMARA HİBRİT TAHMİN BOTU")
    print("   22'lik Model (40-80) + Rastgele = En İyisi")
    print("   HİLE YOK - SADECE GEÇMİŞ VERİLER")
    print("=" * 60)
    
    bot = HybridPredictor()
    bot.load_data()
    
    # Optimizasyon yap (hile yok - geçmiş verilerle)
    best_config = bot.optimize(test_size=50)
    
    # Raporu göster
    prediction = bot.print_report(best_config)
    
    # Sonuçları kaydet
    os.makedirs('outputs', exist_ok=True)
    
    results = {
        'prediction': prediction,
        'best_config': best_config,
        'data_range': {
            'start': bot.df['tarih'].min().strftime('%d.%m.%Y'),
            'end': bot.df['tarih'].max().strftime('%d.%m.%Y'),
            'total_draws': len(bot.df)
        }
    }
    
    with open('outputs/hybrid_predictions.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    with open('outputs/hybrid_report.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("🎯 ON NUMARA HİBRİT TAHMİN RAPORU\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Son Çekiliş: {bot.df['tarih'].max().strftime('%d.%m.%Y')}\n")
        f.write(f"Önerilen 16 Sayı: {prediction['final_numbers']}\n\n")
        f.write(f"Bot Sayıları (40-80): {prediction['bot_numbers']}\n")
        f.write(f"Rastgele Sayılar: {prediction['random_numbers'][:5]}...\n")
    
    print(f"\n💾 Kaydedildi: outputs/hybrid_predictions.json")
    print(f"💾 Kaydedildi: outputs/hybrid_report.txt")
    print("\n✅ TAMAMLANDI! (Hile yok - söz!)")

if __name__ == "__main__":
    main()
