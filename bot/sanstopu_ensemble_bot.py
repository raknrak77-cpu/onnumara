#!/usr/bin/env python3
"""
ŞANS TOPU ENSEMBLE BOTU
En iyi 5 pattern'in önerdiği sayıları birleştirir
Hangi sayıların çıkma olasılığı yüksek listeler
"""

import pandas as pd
import numpy as np
from collections import Counter
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# VERİ YÜKLEYİCİ
# ============================================================

class DataLoader:
    def __init__(self, excel_path="sanstopu.xlsx", sheet_name="s1"):
        self.excel_path = excel_path
        self.sheet_name = sheet_name
        self.df = None
        self.main_columns = ['no_1', 'no_2', 'no_3', 'no_4', 'no_5']
        self.plus_column = 'no_5+1'
        
    def load(self):
        self.df = pd.read_excel(self.excel_path, sheet_name=self.sheet_name, header=0)
        
        if 'tarih' in self.df.columns:
            self.df['tarih'] = pd.to_datetime(self.df['tarih'], format='%d/%m/%Y', errors='coerce')
        elif 'tarih.1' in self.df.columns:
            self.df['tarih'] = pd.to_datetime(self.df['tarih.1'], format='%d/%m/%Y', errors='coerce')
        
        for col in self.main_columns + [self.plus_column]:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        self.df = self.df.dropna(subset=['tarih'] + self.main_columns + [self.plus_column], how='any')
        self.df = self.df.reset_index(drop=True)
        
        self.df['weekday'] = self.df['tarih'].dt.weekday
        self.df['year'] = self.df['tarih'].dt.year
        self.df['day'] = self.df['tarih'].dt.day
        
        return self.df
    
    def get_main_numbers(self, row):
        nums = []
        for col in self.main_columns:
            if col in row.index and pd.notna(row[col]):
                try:
                    nums.append(int(row[col]))
                except:
                    pass
        return nums
    
    def get_plus_number(self, row):
        if self.plus_column in row.index and pd.notna(row[self.plus_column]):
            try:
                return int(row[self.plus_column])
            except:
                return 0
        return 0


# ============================================================
# EN İYİ 5 PATTERN (BACKTEST SONUÇLARINA GÖRE)
# ============================================================

class TopPatterns:
    def __init__(self, df, get_main_func, get_plus_func):
        self.df = df
        self.get_main = get_main_func
        self.get_plus = get_plus_func
    
    # ===== ANA KISIM PATTERNLERİ (En iyi 5) =====
    
    # Pattern 1: Fibonacci + komşu (1.70/10)
    def pattern_main_fibonacci_neighbor(self, k=12):
        fib = [1, 2, 3, 5, 8, 13, 21, 34]
        neighbors = set()
        for f in fib:
            neighbors.add(f)
            if f > 1: neighbors.add(f-1)
            if f < 34: neighbors.add(f+1)
        neighbors = sorted(list(neighbors))
        
        # Sıklığa göre sırala
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_main(row))
        counter = Counter(all_nums)
        neighbors.sort(key=lambda x: counter.get(x, 0), reverse=True)
        return neighbors[:k]
    
    # Pattern 2: Fibonacci aralığı (1.67/10)
    def pattern_main_fibonacci_range(self, k=12):
        fib = [1, 2, 3, 5, 8, 13, 21, 34]
        ranges = set()
        for i in range(len(fib)-1):
            for x in range(fib[i], fib[i+1]+1):
                if x <= 34:
                    ranges.add(x)
        ranges = sorted(list(ranges))
        
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_main(row))
        counter = Counter(all_nums)
        ranges.sort(key=lambda x: counter.get(x, 0), reverse=True)
        return ranges[:k]
    
    # Pattern 3: Tek numaralı çekilişler (1.66/10)
    def pattern_main_odd_draws(self, k=12):
        odd_indices = self.df.iloc[1::2]
        all_nums = []
        for _, row in odd_indices.iterrows():
            all_nums.extend(self.get_main(row))
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]
    
    # Pattern 4: Son 5 çekiliş (1.63/10)
    def pattern_main_last5(self, k=12):
        recent_nums = []
        window = min(5, len(self.df))
        for idx in range(len(self.df) - window, len(self.df)):
            recent_nums.extend(self.get_main(self.df.iloc[idx]))
        counter = Counter(recent_nums)
        return [num for num, _ in counter.most_common(k)]
    
    # Pattern 5: Son 30 çekiliş (1.56/10)
    def pattern_main_last30(self, k=12):
        recent_nums = []
        window = min(30, len(self.df))
        for idx in range(len(self.df) - window, len(self.df)):
            recent_nums.extend(self.get_main(self.df.iloc[idx]))
        counter = Counter(recent_nums)
        return [num for num, _ in counter.most_common(k)]
    
    # ===== ARTI KISIM PATTERNLERİ (En iyi 5) =====
    
    # Pattern 1: Son 5 çekiliş (0.29/3)
    def pattern_plus_last5(self, k=4):
        recent_nums = []
        window = min(5, len(self.df))
        for idx in range(len(self.df) - window, len(self.df)):
            num = self.get_plus(self.df.iloc[idx])
            if num > 0:
                recent_nums.append(num)
        counter = Counter(recent_nums)
        return [num for num, _ in counter.most_common(k)]
    
    # Pattern 2: Son 10 çekiliş (0.29/3)
    def pattern_plus_last10(self, k=4):
        recent_nums = []
        window = min(10, len(self.df))
        for idx in range(len(self.df) - window, len(self.df)):
            num = self.get_plus(self.df.iloc[idx])
            if num > 0:
                recent_nums.append(num)
        counter = Counter(recent_nums)
        return [num for num, _ in counter.most_common(k)]
    
    # Pattern 3: Son 7 çekiliş (0.28/3)
    def pattern_plus_last7(self, k=4):
        recent_nums = []
        window = min(7, len(self.df))
        for idx in range(len(self.df) - window, len(self.df)):
            num = self.get_plus(self.df.iloc[idx])
            if num > 0:
                recent_nums.append(num)
        counter = Counter(recent_nums)
        return [num for num, _ in counter.most_common(k)]
    
    # Pattern 4: Son 15 çekiliş (0.28/3)
    def pattern_plus_last15(self, k=4):
        recent_nums = []
        window = min(15, len(self.df))
        for idx in range(len(self.df) - window, len(self.df)):
            num = self.get_plus(self.df.iloc[idx])
            if num > 0:
                recent_nums.append(num)
        counter = Counter(recent_nums)
        return [num for num, _ in counter.most_common(k)]
    
    # Pattern 5: Son 3 çekiliş (0.24/3)
    def pattern_plus_last3(self, k=4):
        recent_nums = []
        window = min(3, len(self.df))
        for idx in range(len(self.df) - window, len(self.df)):
            num = self.get_plus(self.df.iloc[idx])
            if num > 0:
                recent_nums.append(num)
        counter = Counter(recent_nums)
        return [num for num, _ in counter.most_common(k)]


# ============================================================
# ENSEMBLE BOT
# ============================================================

class EnsembleBot:
    def __init__(self, excel_path="sanstopu.xlsx", sheet_name="s1"):
        self.excel_path = excel_path
        self.sheet_name = sheet_name
        self.df = None
        
    def load_data(self):
        loader = DataLoader(self.excel_path, self.sheet_name)
        self.df = loader.load()
        self.get_main = loader.get_main_numbers
        self.get_plus = loader.get_plus_number
        print(f"✅ Veri yüklendi: {len(self.df)} çekiliş")
        if len(self.df) > 0:
            print(f"📅 Son Çekiliş: {self.df['tarih'].max().strftime('%d.%m.%Y')}")
        return self.df
    
    def get_ensemble_predictions(self):
        """En iyi 5 pattern'in önerdiği tüm sayıları birleştir"""
        
        patterns = TopPatterns(self.df, self.get_main, self.get_plus)
        
        # ANA KISIM: Her pattern'den 12 sayı al (toplam 60)
        p1_main = patterns.pattern_main_fibonacci_neighbor(12)
        p2_main = patterns.pattern_main_fibonacci_range(12)
        p3_main = patterns.pattern_main_odd_draws(12)
        p4_main = patterns.pattern_main_last5(12)
        p5_main = patterns.pattern_main_last30(12)
        
        # ARTI KISIM: Her pattern'den 4 sayı al (toplam 20)
        p1_plus = patterns.pattern_plus_last5(4)
        p2_plus = patterns.pattern_plus_last10(4)
        p3_plus = patterns.pattern_plus_last7(4)
        p4_plus = patterns.pattern_plus_last15(4)
        p5_plus = patterns.pattern_plus_last3(4)
        
        # Tüm ana sayıları birleştir ve frekanslarını hesapla
        all_main = p1_main + p2_main + p3_main + p4_main + p5_main
        main_counter = Counter(all_main)
        
        # Frekansa göre sırala
        sorted_main = sorted(main_counter.items(), key=lambda x: x[1], reverse=True)
        
        # Her sayının hangi pattern'lerde olduğunu bul
        detailed_main = []
        for num, freq in sorted_main:
            patterns_list = []
            if num in p1_main: patterns_list.append('Fib+Komşu')
            if num in p2_main: patterns_list.append('FibAralık')
            if num in p3_main: patterns_list.append('TekÇek')
            if num in p4_main: patterns_list.append('Son5')
            if num in p5_main: patterns_list.append('Son30')
            
            detailed_main.append({
                'number': num,
                'frequency': freq,
                'patterns': patterns_list,
                'pattern_count': len(patterns_list)
            })
        
        # ARTI KISIM için aynı işlem
        all_plus = p1_plus + p2_plus + p3_plus + p4_plus + p5_plus
        plus_counter = Counter(all_plus)
        sorted_plus = sorted(plus_counter.items(), key=lambda x: x[1], reverse=True)
        
        detailed_plus = []
        for num, freq in sorted_plus:
            patterns_list = []
            if num in p1_plus: patterns_list.append('Son5')
            if num in p2_plus: patterns_list.append('Son10')
            if num in p3_plus: patterns_list.append('Son7')
            if num in p4_plus: patterns_list.append('Son15')
            if num in p5_plus: patterns_list.append('Son3')
            
            detailed_plus.append({
                'number': num,
                'frequency': freq,
                'patterns': patterns_list,
                'pattern_count': len(patterns_list)
            })
        
        return detailed_main, detailed_plus, p1_main, p2_main, p3_main, p4_main, p5_main, p1_plus, p2_plus, p3_plus, p4_plus, p5_plus
    
    def print_report(self):
        result = self.get_ensemble_predictions()
        detailed_main, detailed_plus, p1_main, p2_main, p3_main, p4_main, p5_main, p1_plus, p2_plus, p3_plus, p4_plus, p5_plus = result
        
        print("\n" + "=" * 70)
        print("🎯 ŞANS TOPU ENSEMBLE BOTU")
        print("   En iyi 5 pattern birleştirildi")
        print("   Toplam 5×12 = 60 ana sayı havuzu")
        print("   Toplam 5×4 = 20 artı sayı havuzu")
        print("=" * 70)
        
        print("\n📊 5 PATTERN'İN ÖNERDİĞİ ANA SAYILAR:")
        print("-" * 70)
        print(f"  Pattern 1 (Fib+Komşu)          : {p1_main[:8]}...")
        print(f"  Pattern 2 (Fib Aralık)         : {p2_main[:8]}...")
        print(f"  Pattern 3 (Tek Çekilişler)     : {p3_main[:8]}...")
        print(f"  Pattern 4 (Son 5 çekiliş)      : {p4_main[:8]}...")
        print(f"  Pattern 5 (Son 30 çekiliş)     : {p5_main[:8]}...")
        
        print("\n📊 5 PATTERN'İN ÖNERDİĞİ ARTI SAYILAR:")
        print("-" * 70)
        print(f"  Pattern 1 (Son 5 çekiliş)      : {p1_plus}")
        print(f"  Pattern 2 (Son 10 çekiliş)     : {p2_plus}")
        print(f"  Pattern 3 (Son 7 çekiliş)      : {p3_plus}")
        print(f"  Pattern 4 (Son 15 çekiliş)     : {p4_plus}")
        print(f"  Pattern 5 (Son 3 çekiliş)      : {p5_plus}")
        
        print("\n" + "-" * 70)
        print("🏆 TÜM ANA SAYILAR (Frekans sıralı)")
        print("-" * 70)
        print(f"\n  {'Sayı':>5} | {'Görülme':>8} | {'Pattern Sayısı':>15} | {'Patternler'}")
        print("  " + "-" * 65)
        
        for item in detailed_main[:20]:
            print(f"  {item['number']:5d} | {item['frequency']:8d} | {item['pattern_count']:15d} | {', '.join(item['patterns'])}")
        
        # En az 3 pattern'de ortak olan ana sayılar
        common_main_3 = [item for item in detailed_main if item['pattern_count'] >= 3]
        
        print("\n" + "-" * 70)
        print("🎯 ANA KISIM - EN GÜÇLÜ ADAYLAR (En az 3 pattern'de ortak)")
        print("-" * 70)
        
        if common_main_3:
            print(f"\n  {len(common_main_3)} sayı: {[c['number'] for c in common_main_3]}")
        else:
            print("\n  (3 pattern'de ortak sayı bulunamadı)")
        
        # En az 2 pattern'de ortak olan ana sayılar
        common_main_2 = [item for item in detailed_main if item['pattern_count'] >= 2]
        
        print("\n" + "-" * 70)
        print("⭐ ANA KISIM - GÜÇLÜ ADAYLAR (En az 2 pattern'de ortak)")
        print("-" * 70)
        
        if common_main_2:
            print(f"\n  {len(common_main_2)} sayı: {[c['number'] for c in common_main_2[:20]]}")
        else:
            print("\n  (2 pattern'de ortak sayı bulunamadı)")
        
        # ARTI KISIM için güçlü adaylar
        common_plus_2 = [item for item in detailed_plus if item['pattern_count'] >= 2]
        common_plus_3 = [item for item in detailed_plus if item['pattern_count'] >= 3]
        
        print("\n" + "-" * 70)
        print("🎯 ARTI KISIM - EN GÜÇLÜ ADAYLAR")
        print("-" * 70)
        
        if common_plus_3:
            print(f"\n  En az 3 pattern'de: {[c['number'] for c in common_plus_3]}")
        elif common_plus_2:
            print(f"\n  En az 2 pattern'de: {[c['number'] for c in common_plus_2]}")
        else:
            print(f"\n  En sık görülenler: {[c['number'] for c in detailed_plus[:4]]}")
        
        print("\n" + "-" * 70)
        print("📊 ARTI SAYILAR (Frekans sıralı)")
        print("-" * 70)
        print(f"\n  {'Sayı':>5} | {'Görülme':>8} | {'Pattern Sayısı':>15} | {'Patternler'}")
        print("  " + "-" * 65)
        
        for item in detailed_plus:
            print(f"  {item['number']:5d} | {item['frequency']:8d} | {item['pattern_count']:15d} | {', '.join(item['patterns'])}")
        
        # TAHMİNLER
        # En çok görülen 10 ana sayı
        top_10_main = [item['number'] for item in detailed_main[:10]]
        
        print("\n" + "-" * 70)
        print("🎯 ÖNERİLEN 10 ANA SAYI (En yüksek frekans)")
        print("-" * 70)
        
        for i in range(0, len(top_10_main), 5):
            group = top_10_main[i:i+5]
            print(f"  {i+1:2d}-{i+5:2d}. {' '.join(f'{n:3d}' for n in group)}")
        
        # En çok görülen 5 ana sayı
        top_5_main = [item['number'] for item in detailed_main[:5]]
        
        print("\n" + "-" * 70)
        print("🔥 ÖNERİLEN 5 ANA SAYI (En güçlü)")
        print("-" * 70)
        print(f"\n  🌟🌟🌟  {top_5_main}  🌟🌟🌟")
        
        # ARTI TAHMİN - en çok görülen 2 artı sayı
        top_2_plus = [item['number'] for item in detailed_plus[:2]]
        
        print("\n" + "-" * 70)
        print("🔥 ÖNERİLEN ARTI SAYILAR")
        print("-" * 70)
        print(f"\n  🌟  {top_2_plus}  🌟")
        
        # KOMBİNASYON ÖNERİSİ
        print("\n" + "-" * 70)
        print("🎰 ÖNERİLEN KOMBİNASYONLAR")
        print("-" * 70)
        
        # En güçlü 5 ana + en güçlü 2 artı
        combo1_main = top_5_main[:5]
        combo1_plus = top_2_plus[:1] if top_2_plus else [1]
        
        print(f"\n  Kombinasyon 1 (En güçlü 5+1):")
        print(f"    Ana: {sorted(combo1_main)}")
        print(f"    Artı: {combo1_plus[0]}")
        
        # 2. kombinasyon: En çok pattern'de ortak olanlar (varsa)
        if common_main_3:
            combo2_main = [c['number'] for c in common_main_3[:5]]
            combo2_plus = common_plus_3[0]['number'] if common_plus_3 else (common_plus_2[0]['number'] if common_plus_2 else top_2_plus[0])
            print(f"\n  Kombinasyon 2 (En çok pattern ortaklığı):")
            print(f"    Ana: {sorted(combo2_main)}")
            print(f"    Artı: {combo2_plus}")
        
        # 3. kombinasyon: Frekans 2+ olanlar
        if common_main_2:
            combo3_main = [c['number'] for c in common_main_2[:5]]
            combo3_plus = common_plus_2[0]['number'] if common_plus_2 else top_2_plus[0]
            print(f"\n  Kombinasyon 3 (En az 2 pattern'de ortak):")
            print(f"    Ana: {sorted(combo3_main)}")
            print(f"    Artı: {combo3_plus}")
        
        # İSTATİSTİK
        print("\n" + "-" * 70)
        print("📊 İSTATİSTİK")
        print("-" * 70)
        
        total_unique_main = len(detailed_main)
        print(f"  Toplam farklı ana sayı: {total_unique_main}/34")
        print(f"  Ortalama her ana sayı: {sum(c['frequency'] for c in detailed_main)/total_unique_main:.2f} pattern'de")
        print(f"  Maksimum ana pattern: {max(c['pattern_count'] for c in detailed_main)}")
        
        print(f"\n  Toplam farklı artı sayı: {len(detailed_plus)}/14")
        if detailed_plus:
            print(f"  Maksimum artı pattern: {max(c['pattern_count'] for c in detailed_plus)}")
        
        print("\n" + "-" * 70)
        print("⚠️ NOT: 5 pattern'in ortak önerdiği sayılar listelenmiştir.")
        print("   Kesin sonuç garantisi yoktur. Eğlence amaçlıdır.")
        print("=" * 70)
        
        return {
            'top_10_main': top_10_main,
            'top_5_main': top_5_main,
            'top_2_plus': top_2_plus,
            'all_main': detailed_main,
            'all_plus': detailed_plus,
            'common_main_3': [c['number'] for c in common_main_3] if common_main_3 else [],
            'common_main_2': [c['number'] for c in common_main_2] if common_main_2 else [],
            'common_plus_2': [c['number'] for c in common_plus_2] if common_plus_2 else [],
            'common_plus_3': [c['number'] for c in common_plus_3] if common_plus_3 else []
        }
    
    def save_results(self, result):
        os.makedirs('outputs', exist_ok=True)
        
        with open('outputs/sanstopu_ensemble_predictions.json', 'w', encoding='utf-8') as f:
            json.dump({
                'recommended_10_main': result['top_10_main'],
                'recommended_5_main': result['top_5_main'],
                'recommended_2_plus': result['top_2_plus'],
                'common_in_3_patterns_main': result['common_main_3'],
                'common_in_2_patterns_main': result['common_main_2'],
                'common_in_2_patterns_plus': result['common_plus_2'],
                'common_in_3_patterns_plus': result['common_plus_3'],
                'all_main_with_frequency': result['all_main'][:20],
                'all_plus_with_frequency': result['all_plus']
            }, f, ensure_ascii=False, indent=2)
        
        with open('outputs/sanstopu_ensemble_report.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("🎯 ŞANS TOPU ENSEMBLE BOT RAPORU\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Son Çekiliş: {self.df['tarih'].max().strftime('%d.%m.%Y')}\n\n")
            f.write("ÖNERİLEN 10 ANA SAYI:\n")
            f.write(str(result['top_10_main']) + "\n\n")
            f.write("ÖNERİLEN 5 ANA SAYI:\n")
            f.write(str(result['top_5_main']) + "\n\n")
            f.write("ÖNERİLEN ARTI SAYILAR:\n")
            f.write(str(result['top_2_plus']) + "\n\n")
            f.write("ANA KISIM - EN AZ 3 PATTERN'DE ORTAK:\n")
            f.write(str(result['common_main_3']) + "\n\n")
            f.write("ANA KISIM - EN AZ 2 PATTERN'DE ORTAK:\n")
            f.write(str(result['common_main_2']) + "\n\n")
            f.write("ARTI KISIM - EN AZ 2 PATTERN'DE ORTAK:\n")
            f.write(str(result['common_plus_2']) + "\n\n")
            f.write("TÜM ANA SAYILAR (Frekans sıralı):\n")
            for item in result['all_main'][:20]:
                f.write(f"  {item['number']:3d}: {item['frequency']} pattern'de ({', '.join(item['patterns'])})\n")
            f.write("\nTÜM ARTI SAYILAR (Frekans sıralı):\n")
            for item in result['all_plus']:
                f.write(f"  {item['number']:3d}: {item['frequency']} pattern'de ({', '.join(item['patterns'])})\n")
        
        print(f"\n💾 Kaydedildi: outputs/sanstopu_ensemble_predictions.json")
        print(f"💾 Kaydedildi: outputs/sanstopu_ensemble_report.txt")


# ============================================================
# ANA ÇALIŞTIRMA
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("🚀 ŞANS TOPU ENSEMBLE BOTU")
    print("   En iyi 5 pattern birleştirildi")
    print("   Ana: 5×12 = 60 sayı havuzu")
    print("   Artı: 5×4 = 20 sayı havuzu")
    print("=" * 70)
    
    bot = EnsembleBot()
    bot.load_data()
    result = bot.print_report()
    bot.save_results(result)
    
    print("\n✅ TAMAMLANDI!")

if __name__ == "__main__":
    main()
