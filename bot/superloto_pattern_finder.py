#!/usr/bin/env python3
"""
SÜPER LOTO PATTERN FINDER BOTU
903 çekiliş içinde PATTERN ara
Geriye doğru 150 çekilişi test ederek en iyi stratejiyi bulur
"""

import pandas as pd
import numpy as np
from collections import Counter
import json
import os
from itertools import combinations

# ============================================================
# VERİ YÜKLEYİCİ
# ============================================================

class DataLoader:
    def __init__(self, excel_path="superloto.xlsx", sheet_name="s1"):
        self.excel_path = excel_path
        self.sheet_name = sheet_name
        self.df = None
        self.number_columns = ['no_1', 'no_2', 'no_3', 'no_4', 'no_5', 'no_6']
        
    def load(self):
        self.df = pd.read_excel(self.excel_path, sheet_name=self.sheet_name, header=0)
        
        if 'tarih' in self.df.columns:
            self.df['tarih'] = pd.to_datetime(self.df['tarih'], format='%d/%m/%Y', errors='coerce')
        
        for col in self.number_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        self.df = self.df.dropna(subset=['tarih'] + self.number_columns, how='any')
        self.df = self.df.reset_index(drop=True)
        
        # Haftanın günü ve ay bilgilerini ekle
        self.df['weekday'] = self.df['tarih'].dt.weekday
        self.df['month'] = self.df['tarih'].dt.month
        self.df['day'] = self.df['tarih'].dt.day
        self.df['year'] = self.df['tarih'].dt.year
        
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
# PATTERN STRATEJİLERİ
# ============================================================

class Patterns:
    def __init__(self, df, get_numbers_func):
        self.df = df
        self.get_numbers = get_numbers_func
    
    # 1. STRATEJİ: Son N çekilişte en sık çıkan sayılar
    def strategy_recent(self, n=12, window=5):
        recent_nums = []
        window = min(window, len(self.df))
        for idx in range(len(self.df) - window, len(self.df)):
            recent_nums.extend(self.get_numbers(self.df.iloc[idx]))
        counter = Counter(recent_nums)
        return [num for num, _ in counter.most_common(n)]
    
    # 2. STRATEJİ: En uzun süredir çıkmayanlar (Due)
    def strategy_due(self, n=12):
        last_seen = {num: 0 for num in range(1, 61)}
        for idx, row in self.df.iterrows():
            for num in self.get_numbers(row):
                last_seen[num] = idx
        last_idx = len(self.df) - 1
        due_counts = {num: last_idx - last_seen[num] for num in range(1, 61)}
        due_sorted = sorted(due_counts.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in due_sorted[:n]]
    
    # 3. STRATEJİ: Son çekilişteki sayıların +1/-1'i
    def strategy_neighbors(self, n=12):
        last_row = self.df.iloc[-1]
        last_nums = self.get_numbers(last_row)
        neighbors = set()
        for num in last_nums:
            if num > 1:
                neighbors.add(num - 1)
            if num < 60:
                neighbors.add(num + 1)
            neighbors.add(num)
        result = list(neighbors)
        while len(result) < n:
            result.append(np.random.randint(1, 61))
        return result[:n]
    
    # 4. STRATEJİ: Tek/Çift dengesi (3 tek + 3 çift)
    def strategy_odd_even(self, n=12):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        
        odds = [num for num in range(1, 61) if num % 2 == 1]
        evens = [num for num in range(1, 61) if num % 2 == 0]
        
        odd_counts = Counter([num for num in all_nums if num % 2 == 1])
        even_counts = Counter([num for num in all_nums if num % 2 == 0])
        
        top_odds = [num for num, _ in odd_counts.most_common(6)]
        top_evens = [num for num, _ in even_counts.most_common(6)]
        
        return (top_odds + top_evens)[:n]
    
    # 5. STRATEJİ: Büyük/Küçük dengesi (1-30 / 31-60)
    def strategy_small_large(self, n=12):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        
        small = [num for num in range(1, 31)]
        large = [num for num in range(31, 61)]
        
        small_counts = Counter([num for num in all_nums if num <= 30])
        large_counts = Counter([num for num in all_nums if num > 30])
        
        top_small = [num for num, _ in small_counts.most_common(6)]
        top_large = [num for num, _ in large_counts.most_common(6)]
        
        return (top_small + top_large)[:n]
    
    # 6. STRATEJİ: Haftanın gününe göre filtreleme
    def strategy_weekday(self, n=12):
        last_weekday = self.df.iloc[-1]['weekday']
        same_weekday = self.df[self.df['weekday'] == last_weekday]
        
        all_nums = []
        for _, row in same_weekday.iterrows():
            all_nums.extend(self.get_numbers(row))
        
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(n)]
    
    # 7. STRATEJİ: Ayın gününe göre
    def strategy_month_day(self, n=12):
        last_day = self.df.iloc[-1]['day']
        same_day = self.df[self.df['day'] == last_day]
        
        if len(same_day) < 5:
            return self.strategy_recent(n)
        
        all_nums = []
        for _, row in same_day.iterrows():
            all_nums.extend(self.get_numbers(row))
        
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(n)]
    
    # 8. STRATEJİ: Kombinasyon (Son çekiliş + en sık)
    def strategy_hybrid(self, n=12):
        recent = self.strategy_recent(n*2, 5)
        due = self.strategy_due(n*2)
        
        all_votes = recent + due
        counter = Counter(all_votes)
        return [num for num, _ in counter.most_common(n)]
    
    # 9. STRATEJİ: Son 3 çekilişin en sık sayıları
    def strategy_last3(self, n=12):
        last3_nums = []
        for idx in range(max(0, len(self.df)-3), len(self.df)):
            last3_nums.extend(self.get_numbers(self.df.iloc[idx]))
        counter = Counter(last3_nums)
        return [num for num, _ in counter.most_common(n)]
    
    # 10. STRATEJİ: Asal sayı ağırlıklı
    def strategy_prime(self, n=12):
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]
        
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        
        prime_counts = Counter([num for num in all_nums if num in primes])
        nonprime_counts = Counter([num for num in all_nums if num not in primes])
        
        top_primes = [num for num, _ in prime_counts.most_common(6)]
        top_nonprimes = [num for num, _ in nonprime_counts.most_common(6)]
        
        return (top_primes + top_nonprimes)[:n]


# ============================================================
# BACKTEST ve OPTİMİZASYON
# ============================================================

class PatternTester:
    def __init__(self, df, get_numbers_func):
        self.df = df
        self.get_numbers = get_numbers_func
        self.patterns = Patterns(df, get_numbers_func)
        self.results = {}
        
    def test_strategy(self, strategy_func, strategy_name, test_size=150):
        """Bir stratejiyi backtest et"""
        total = len(self.df)
        train_size = total - test_size
        
        if train_size <= 0:
            return 0
        
        scores = []
        for i in range(test_size):
            train_end = train_size + i
            if train_end >= total:
                break
            
            # Geçici veri ile stratejiyi çalıştır
            temp_df = self.df.iloc[:train_end]
            temp_patterns = Patterns(temp_df, self.get_numbers)
            
            if strategy_name == 'recent':
                preds = temp_patterns.strategy_recent(12, 5)
            elif strategy_name == 'due':
                preds = temp_patterns.strategy_due(12)
            elif strategy_name == 'neighbors':
                preds = temp_patterns.strategy_neighbors(12)
            elif strategy_name == 'odd_even':
                preds = temp_patterns.strategy_odd_even(12)
            elif strategy_name == 'small_large':
                preds = temp_patterns.strategy_small_large(12)
            elif strategy_name == 'weekday':
                preds = temp_patterns.strategy_weekday(12)
            elif strategy_name == 'month_day':
                preds = temp_patterns.strategy_month_day(12)
            elif strategy_name == 'hybrid':
                preds = temp_patterns.strategy_hybrid(12)
            elif strategy_name == 'last3':
                preds = temp_patterns.strategy_last3(12)
            elif strategy_name == 'prime':
                preds = temp_patterns.strategy_prime(12)
            else:
                preds = []
            
            test_row = self.df.iloc[train_end]
            actual = set(self.get_numbers(test_row))
            correct = len(set(preds) & actual)
            scores.append(correct)
        
        avg_score = sum(scores) / len(scores) if scores else 0
        return avg_score
    
    def find_best_window_for_recent(self, test_size=150):
        print("  🔍 Son N çekiliş için en iyi pencere aranıyor...")
        best_score = 0
        best_window = 5
        windows = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
        
        for window in windows:
            if window > len(self.df) - test_size:
                continue
            
            scores = []
            total = len(self.df)
            train_size = total - test_size
            
            for i in range(test_size):
                train_end = train_size + i
                if train_end >= total:
                    break
                
                temp_df = self.df.iloc[:train_end]
                temp_patterns = Patterns(temp_df, self.get_numbers)
                preds = temp_patterns.strategy_recent(12, window)
                
                test_row = self.df.iloc[train_end]
                actual = set(self.get_numbers(test_row))
                correct = len(set(preds) & actual)
                scores.append(correct)
            
            avg_score = sum(scores) / len(scores) if scores else 0
            print(f"     window={window:2d} -> {avg_score:.2f}/12 doğru")
            if avg_score > best_score:
                best_score = avg_score
                best_window = window
        
        print(f"  ✅ En iyi pencere: {best_window} (ortalama {best_score:.2f}/12)")
        return best_window, best_score
    
    def run_all_tests(self, test_size=150):
        print(f"\n📊 PATTERN TESTİ BAŞLIYOR (son {test_size} çekiliş üzerinde)")
        print("   Hedef: 60 sayı içinden 12 tahmin")
        print("-" * 50)
        
        strategies = [
            ('recent', 'Son 5 çekiliş'),
            ('due', 'En uzun süredir çıkmayanlar'),
            ('neighbors', 'Son çekilişin komşuları'),
            ('odd_even', 'Tek/Çift dengesi'),
            ('small_large', 'Büyük/Küçük dengesi'),
            ('weekday', 'Haftanın günü'),
            ('month_day', 'Ayın günü'),
            ('hybrid', 'Kombinasyon (recent+due)'),
            ('last3', 'Son 3 çekiliş'),
            ('prime', 'Asal ağırlıklı')
        ]
        
        print("\n  📈 STRATEJİ TEST SONUÇLARI:")
        for strategy_key, strategy_name in strategies:
            score = self.test_strategy(None, strategy_key, test_size)
            self.results[strategy_key] = score
            print(f"     {strategy_name:30}: {score:.2f}/12 doğru")
        
        # Recent için en iyi pencereyi bul
        best_window, recent_score = self.find_best_window_for_recent(test_size)
        self.results['recent_optimized'] = recent_score
        self.results['recent_best_window'] = best_window
        
        print("\n" + "-" * 50)
        print("🏆 STRATEJİ BAŞARI SIRALAMASI (12'de kaç doğru):")
        print("-" * 50)
        
        sorted_results = sorted([(k, v) for k, v in self.results.items() if isinstance(v, (int, float))], key=lambda x: x[1], reverse=True)
        for i, (strategy, score) in enumerate(sorted_results):
            stars = "⭐" * int(score // 1) + "☆" * (6 - int(score // 1))
            print(f"  {i+1}. {strategy:20}: {score:.2f}/12 {stars}")
        
        return self.results


# ============================================================
# ANA SINIF
# ============================================================

class PatternFinder:
    def __init__(self, excel_path="superloto.xlsx", sheet_name="s1"):
        self.excel_path = excel_path
        self.sheet_name = sheet_name
        self.df = None
        self.test_results = None
        self.best_strategy = None
        
    def load_data(self):
        loader = DataLoader(self.excel_path, self.sheet_name)
        self.df = loader.load()
        self.get_numbers = loader.get_numbers
        print(f"✅ Veri yüklendi: {len(self.df)} çekiliş")
        if len(self.df) > 0:
            print(f"📅 Aralık: {self.df['tarih'].min().strftime('%d.%m.%Y')} - {self.df['tarih'].max().strftime('%d.%m.%Y')}")
        return self.df
    
    def run_tests(self, test_size=150):
        tester = PatternTester(self.df, self.get_numbers)
        self.test_results = tester.run_all_tests(test_size)
        
        # En iyi stratejiyi bul
        best = max([(k, v) for k, v in self.test_results.items() if isinstance(v, (int, float))], key=lambda x: x[1])
        self.best_strategy = best[0]
        
        print(f"\n🎯 EN İYİ STRATEJİ: {best[0]} (Ortalama {best[1]:.2f}/12)")
        
        return self.test_results
    
    def get_prediction(self):
        """En iyi strateji ile tahmin yap"""
        patterns = Patterns(self.df, self.get_numbers)
        
        if self.best_strategy == 'recent' or self.best_strategy == 'recent_optimized':
            window = self.test_results.get('recent_best_window', 5)
            preds = patterns.strategy_recent(12, window)
        elif self.best_strategy == 'due':
            preds = patterns.strategy_due(12)
        elif self.best_strategy == 'neighbors':
            preds = patterns.strategy_neighbors(12)
        elif self.best_strategy == 'odd_even':
            preds = patterns.strategy_odd_even(12)
        elif self.best_strategy == 'small_large':
            preds = patterns.strategy_small_large(12)
        elif self.best_strategy == 'weekday':
            preds = patterns.strategy_weekday(12)
        elif self.best_strategy == 'month_day':
            preds = patterns.strategy_month_day(12)
        elif self.best_strategy == 'hybrid':
            preds = patterns.strategy_hybrid(12)
        elif self.best_strategy == 'last3':
            preds = patterns.strategy_last3(12)
        elif self.best_strategy == 'prime':
            preds = patterns.strategy_prime(12)
        else:
            preds = patterns.strategy_recent(12, 5)
        
        # En iyi 6 sayıyı seç (tek/çift dengesine dikkat ederek)
        odd_count = sum(1 for n in preds[:6] if n % 2 == 1)
        if odd_count < 2:
            for i, n in enumerate(preds[6:]):
                if n % 2 == 1 and len(preds[:6]) < 6:
                    preds = [n] + preds
        elif odd_count > 4:
            for i, n in enumerate(preds[6:]):
                if n % 2 == 0 and len(preds[:6]) < 6:
                    preds = [n] + preds
        
        return preds[:12], preds[:6]
    
    def print_report(self):
        if self.test_results is None:
            self.run_tests()
        
        final_12, best_6 = self.get_prediction()
        
        print("\n" + "=" * 60)
        print("🎯 SÜPER LOTO PATTERN FINDER BOTU")
        print("   903 çekiliş analizi + 150 çekiliş testi")
        print("=" * 60)
        print(f"\n📅 Son Çekiliş: {self.df['tarih'].max().strftime('%d.%m.%Y')}")
        print(f"📊 Toplam Analiz: {len(self.df)} çekiliş")
        print(f"🎯 En İyi Strateji: {self.best_strategy}")
        
        print("\n" + "-" * 60)
        print("🏆 EN GÜÇLÜ 12 SAYI (En iyi strateji ile)")
        print("-" * 60)
        
        for i in range(0, len(final_12), 4):
            group = final_12[i:i+4]
            print(f"  {i+1:2d}-{i+4:2d}. {' '.join(f'{n:3d}' for n in group)}")
        
        print("\n" + "-" * 60)
        print("🎯 ÖNERİLEN 6 SAYI")
        print("-" * 60)
        print(f"\n  🌟🌟🌟  {best_6}  🌟🌟🌟")
        
        print("\n" + "-" * 60)
        print("📊 STRATEJİ BAŞARI SIRALAMASI (150 test sonucu)")
        print("-" * 60)
        
        sorted_results = sorted([(k, v) for k, v in self.test_results.items() if isinstance(v, (int, float))], key=lambda x: x[1], reverse=True)
        for i, (strategy, score) in enumerate(sorted_results[:10]):
            print(f"  {i+1}. {strategy:20}: {score:.2f}/12")
        
        print("\n" + "-" * 60)
        print("⚠️ NOT: Bu tahminler istatistiksel analizdir.")
        print("   Kesin sonuç garantisi yoktur. Eğlence amaçlıdır.")
        print("=" * 60)
        
        return {'final_12': final_12, 'best_6': best_6}
    
    def save_results(self, final_12, best_6, filename="pattern_finder_results.json"):
        os.makedirs('outputs', exist_ok=True)
        
        with open(f'outputs/{filename}', 'w', encoding='utf-8') as f:
            json.dump({
                'best_12_numbers': final_12,
                'recommended_6_numbers': best_6,
                'best_strategy': self.best_strategy,
                'test_results': self.test_results
            }, f, ensure_ascii=False, indent=2)
        
        with open('outputs/pattern_finder_report.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("🎯 SÜPER LOTO PATTERN FINDER RAPORU\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Son Çekiliş: {self.df['tarih'].max().strftime('%d.%m.%Y')}\n")
            f.write(f"Toplam Analiz: {len(self.df)} çekiliş\n")
            f.write(f"En İyi Strateji: {self.best_strategy}\n\n")
            f.write("EN GÜÇLÜ 12 SAYI:\n")
            f.write(str(final_12) + "\n\n")
            f.write("🎯 ÖNERİLEN 6 SAYI:\n")
            f.write(str(best_6) + "\n\n")
            f.write("STRATEJİ BAŞARI SIRALAMASI (150 test):\n")
            for k, v in sorted(self.test_results.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True):
                if isinstance(v, (int, float)):
                    f.write(f"  {k}: {v:.2f}/12\n")
        
        print(f"\n💾 Kaydedildi: outputs/{filename}")
        print(f"💾 Kaydedildi: outputs/pattern_finder_report.txt")


# ============================================================
# ANA ÇALIŞTIRMA
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("🚀 SÜPER LOTO PATTERN FINDER BOTU")
    print("   903 çekiliş içinde PATTERN ara")
    print("   Geriye doğru 150 çekiliş test eder")
    print("=" * 60)
    
    bot = PatternFinder()
    bot.load_data()
    bot.run_tests(test_size=150)
    result = bot.print_report()
    bot.save_results(result['final_12'], result['best_6'])
    
    print("\n✅ TAMAMLANDI!")

if __name__ == "__main__":
    main()
