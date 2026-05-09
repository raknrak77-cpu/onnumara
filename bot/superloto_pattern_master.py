#!/usr/bin/env python3
"""
SÜPER LOTO PATTERN MASTER BOTU
25+ pattern test eder, hangisi gerçekten işe yarıyor bulur
903 çekiliş, 150 backtest
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
        
        # Ek bilgiler
        self.df['weekday'] = self.df['tarih'].dt.weekday
        self.df['month'] = self.df['tarih'].dt.month
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
# 35 PATTERN STRATEJİSİ
# ============================================================

class PatternMaster:
    def __init__(self, df, get_numbers_func):
        self.df = df
        self.get_numbers = get_numbers_func
    
    # 1-7: ZAMAN TABANLI
    def p_son_n(self, n, k=12):
        recent_nums = []
        window = min(n, len(self.df))
        for idx in range(len(self.df) - window, len(self.df)):
            recent_nums.extend(self.get_numbers(self.df.iloc[idx]))
        counter = Counter(recent_nums)
        return [num for num, _ in counter.most_common(k)]
    
    def p_son_1(self): return self.p_son_n(1, 12)
    def p_son_3(self): return self.p_son_n(3, 12)
    def p_son_5(self): return self.p_son_n(5, 12)
    def p_son_7(self): return self.p_son_n(7, 12)
    def p_son_10(self): return self.p_son_n(10, 12)
    def p_son_15(self): return self.p_son_n(15, 12)
    def p_son_20(self): return self.p_son_n(20, 12)
    def p_son_30(self): return self.p_son_n(30, 12)
    
    # 8-10: FİBONACCİ
    def p_fibonacci(self, k=12):
        fib = [1, 2, 3, 5, 8, 13, 21, 34, 55]
        return fib[:k]
    
    def p_fibonacci_neighbor(self, k=12):
        fib = [1, 2, 3, 5, 8, 13, 21, 34, 55]
        neighbors = set()
        for f in fib:
            neighbors.add(f)
            if f > 1: neighbors.add(f-1)
            if f < 60: neighbors.add(f+1)
        return sorted(list(neighbors))[:k]
    
    def p_fibonacci_range(self, k=12):
        fib = [1, 2, 3, 5, 8, 13, 21, 34, 55]
        ranges = set()
        for i in range(len(fib)-1):
            for x in range(fib[i], fib[i+1]+1):
                ranges.add(x)
        return sorted(list(ranges))[:k]
    
    # 11-14: TEK/ÇİFT
    def p_only_odd(self, k=12):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        odds = [num for num in all_nums if num % 2 == 1]
        counter = Counter(odds)
        return [num for num, _ in counter.most_common(k)]
    
    def p_only_even(self, k=12):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        evens = [num for num in all_nums if num % 2 == 0]
        counter = Counter(evens)
        return [num for num, _ in counter.most_common(k)]
    
    def p_3odd_3even(self, k=12):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        odds = [num for num in all_nums if num % 2 == 1]
        evens = [num for num in all_nums if num % 2 == 0]
        odd_counter = Counter(odds)
        even_counter = Counter(evens)
        top_odds = [num for num, _ in odd_counter.most_common(6)]
        top_evens = [num for num, _ in even_counter.most_common(6)]
        return (top_odds + top_evens)[:k]
    
    def p_4odd_2even(self, k=12):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        odds = [num for num in all_nums if num % 2 == 1]
        evens = [num for num in all_nums if num % 2 == 0]
        odd_counter = Counter(odds)
        even_counter = Counter(evens)
        top_odds = [num for num, _ in odd_counter.most_common(8)]
        top_evens = [num for num, _ in even_counter.most_common(4)]
        return (top_odds + top_evens)[:k]
    
    # 15-18: BÜYÜK/KÜÇÜK
    def p_small(self, k=12):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        small = [num for num in all_nums if num <= 30]
        counter = Counter(small)
        return [num for num, _ in counter.most_common(k)]
    
    def p_large(self, k=12):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        large = [num for num in all_nums if num > 30]
        counter = Counter(large)
        return [num for num, _ in counter.most_common(k)]
    
    def p_3small_3large(self, k=12):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        small = [num for num in all_nums if num <= 30]
        large = [num for num in all_nums if num > 30]
        small_counter = Counter(small)
        large_counter = Counter(large)
        top_small = [num for num, _ in small_counter.most_common(6)]
        top_large = [num for num, _ in large_counter.most_common(6)]
        return (top_small + top_large)[:k]
    
    # 19-21: MODÜLER
    def p_mod5(self, k=12):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        mod5 = [num for num in all_nums if num % 5 == 0]
        counter = Counter(mod5)
        return [num for num, _ in counter.most_common(k)]
    
    def p_mod10(self, k=12):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        mod10 = [num for num in all_nums if num % 10 == 0]
        counter = Counter(mod10)
        return [num for num, _ in counter.most_common(k)]
    
    def p_mod3(self, k=12):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        mod3 = [num for num in all_nums if num % 3 == 0]
        counter = Counter(mod3)
        return [num for num, _ in counter.most_common(k)]
    
    # 22-24: ÖZEL SAYILAR
    def p_prime(self, k=12):
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]
        actual_primes = []
        for p in primes:
            if p <= 60:
                actual_primes.append(p)
        return actual_primes[:k]
    
    def p_perfect_square(self, k=12):
        squares = [1, 4, 9, 16, 25, 36, 49]
        return squares[:k]
    
    def p_power2(self, k=12):
        powers = [1, 2, 4, 8, 16, 32]
        return powers[:k]
    
    # 25-26: DUE VE HOT
    def p_due(self, k=12):
        last_seen = {num: 0 for num in range(1, 61)}
        for idx, row in self.df.iterrows():
            for num in self.get_numbers(row):
                last_seen[num] = idx
        last_idx = len(self.df) - 1
        due_counts = {num: last_idx - last_seen[num] for num in range(1, 61)}
        due_sorted = sorted(due_counts.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in due_sorted[:k]]
    
    def p_hot(self, k=12):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]
    
    # 27-28: TREND
    def p_trend_up(self, k=12, window=20):
        scores = {}
        for num in range(1, 61):
            recent_window = self.df.iloc[-window:]
            older_window = self.df.iloc[-window*2:-window] if len(self.df) > window*2 else self.df.iloc[:window]
            
            recent_count = sum(1 for _, row in recent_window.iterrows() if num in self.get_numbers(row))
            older_count = sum(1 for _, row in older_window.iterrows() if num in self.get_numbers(row))
            
            if older_count > 0:
                trend = recent_count - older_count
            else:
                trend = recent_count
            scores[num] = trend
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:k]]
    
    def p_trend_down(self, k=12, window=20):
        scores = {}
        for num in range(1, 61):
            recent_window = self.df.iloc[-window:]
            older_window = self.df.iloc[-window*2:-window] if len(self.df) > window*2 else self.df.iloc[:window]
            
            recent_count = sum(1 for _, row in recent_window.iterrows() if num in self.get_numbers(row))
            older_count = sum(1 for _, row in older_window.iterrows() if num in self.get_numbers(row))
            
            if older_count > 0:
                trend = older_count - recent_count
            else:
                trend = 0
            scores[num] = trend
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:k]]
    
    # 29-30: KOMŞULUK
    def p_neighbor(self, k=12):
        last_row = self.df.iloc[-1]
        last_nums = self.get_numbers(last_row)
        neighbors = set()
        for num in last_nums:
            for offset in [-2, -1, 0, 1, 2]:
                neighbor = num + offset
                if 1 <= neighbor <= 60:
                    neighbors.add(neighbor)
        return sorted(list(neighbors))[:k]
    
    def p_last_draw(self, k=12):
        last_row = self.df.iloc[-1]
        return self.get_numbers(last_row)[:k]
    
    # 31: ARALIK ANALİZİ
    def p_interval(self, k=12):
        intervals = []
        for _, row in self.df.iterrows():
            nums = sorted(self.get_numbers(row))
            for i in range(len(nums)-1):
                intervals.append(nums[i+1] - nums[i])
        
        avg_interval = sum(intervals) / len(intervals) if intervals else 5
        last_row = sorted(self.get_numbers(self.df.iloc[-1]))
        predictions = []
        last_num = last_row[-1]
        for _ in range(k):
            next_num = last_num + int(avg_interval)
            if next_num > 60:
                next_num = next_num - 60
            predictions.append(next_num)
            last_num = next_num
        return predictions
    
    # 32: HAFTANIN GÜNÜ
    def p_weekday(self, k=12):
        last_weekday = self.df.iloc[-1]['weekday']
        same_weekday = self.df[self.df['weekday'] == last_weekday]
        all_nums = []
        for _, row in same_weekday.iterrows():
            all_nums.extend(self.get_numbers(row))
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]
    
    # 33: AYIN GÜNÜ
    def p_month_day(self, k=12):
        last_day = self.df.iloc[-1]['day']
        same_day = self.df[self.df['day'] == last_day]
        if len(same_day) < 5:
            return self.p_hot(k)
        all_nums = []
        for _, row in same_day.iterrows():
            all_nums.extend(self.get_numbers(row))
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]
    
    # 34-35: ÇİFT/TEK ÇEKİLİŞLER
    def p_even_draws(self, k=12):
        even_indices = self.df.iloc[::2]  # çift numaralı çekilişler
        all_nums = []
        for _, row in even_indices.iterrows():
            all_nums.extend(self.get_numbers(row))
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]
    
    def p_odd_draws(self, k=12):
        odd_indices = self.df.iloc[1::2]  # tek numaralı çekilişler
        all_nums = []
        for _, row in odd_indices.iterrows():
            all_nums.extend(self.get_numbers(row))
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]


# ============================================================
# BACKTEST MOTORU
# ============================================================

class BacktestEngine:
    def __init__(self, df, get_numbers_func):
        self.df = df
        self.get_numbers = get_numbers_func
        self.patterns = PatternMaster(df, get_numbers_func)
        self.results = {}
        
    def test_pattern(self, pattern_func, pattern_name, test_size=150):
        """Tek bir pattern'i backtest et"""
        total = len(self.df)
        train_size = total - test_size
        
        if train_size <= 0:
            return 0
        
        scores = []
        for i in range(test_size):
            train_end = train_size + i
            if train_end >= total:
                break
            
            temp_df = self.df.iloc[:train_end]
            temp_patterns = PatternMaster(temp_df, self.get_numbers)
            
            try:
                preds = getattr(temp_patterns, pattern_func)()
            except:
                preds = []
            
            test_row = self.df.iloc[train_end]
            actual = set(self.get_numbers(test_row))
            correct = len(set(preds) & actual)
            scores.append(correct)
        
        avg_score = sum(scores) / len(scores) if scores else 0
        return avg_score
    
    def run_all_tests(self, test_size=150):
        print(f"\n📊 PATTERN TESTİ BAŞLIYOR (son {test_size} çekiliş üzerinde)")
        print("   Toplam 35 pattern test edilecek")
        print("-" * 60)
        
        pattern_list = [
            # Zaman tabanlı
            ('p_son_1', 'Son 1 çekiliş'),
            ('p_son_3', 'Son 3 çekiliş'),
            ('p_son_5', 'Son 5 çekiliş'),
            ('p_son_7', 'Son 7 çekiliş'),
            ('p_son_10', 'Son 10 çekiliş'),
            ('p_son_15', 'Son 15 çekiliş'),
            ('p_son_20', 'Son 20 çekiliş'),
            ('p_son_30', 'Son 30 çekiliş'),
            # Fibonacci
            ('p_fibonacci', 'Fibonacci sayıları'),
            ('p_fibonacci_neighbor', 'Fibonacci + komşu'),
            ('p_fibonacci_range', 'Fibonacci aralığı'),
            # Tek/Çift
            ('p_only_odd', 'Sadece tek sayılar'),
            ('p_only_even', 'Sadece çift sayılar'),
            ('p_3odd_3even', '3 tek + 3 çift'),
            ('p_4odd_2even', '4 tek + 2 çift'),
            # Büyük/Küçük
            ('p_small', 'Küçük sayılar (1-30)'),
            ('p_large', 'Büyük sayılar (31-60)'),
            ('p_3small_3large', '3 küçük + 3 büyük'),
            # Modüler
            ('p_mod5', "Mod 5 (5'in katları)"),
            ('p_mod10', "Mod 10 (10'un katları)"),
            ('p_mod3', "Mod 3 (3'ün katları)"),
            # Özel sayılar
            ('p_prime', 'Asal sayılar'),
            ('p_perfect_square', 'Tam kareler'),
            ('p_power2', "2'nin kuvvetleri"),
            # Due & Hot
            ('p_due', 'Due numbers (en uzun süredir çıkmayan)'),
            ('p_hot', 'Hot numbers (en çok çıkan)'),
            # Trend
            ('p_trend_up', 'Trend artan sayılar'),
            ('p_trend_down', 'Trend azalan sayılar'),
            # Komşuluk
            ('p_neighbor', 'Son çekilişin komşuları (±1,±2)'),
            ('p_last_draw', 'Son çekilişin aynısı'),
            ('p_interval', 'Aralık analizi'),
            # Zaman
            ('p_weekday', 'Haftanın aynı günü'),
            ('p_month_day', 'Ayın aynı günü'),
            ('p_even_draws', 'Çift numaralı çekilişler'),
            ('p_odd_draws', 'Tek numaralı çekilişler'),
        ]
        
        print("\n  📈 PATTERN TEST SONUÇLARI:")
        print(f"  {'Pattern':35} {'Başarı':>10}")
        print("  " + "-" * 47)
        
        for pattern_func, pattern_name in pattern_list:
            score = self.test_pattern(pattern_func, pattern_name, test_size)
            self.results[pattern_name] = score
            star = "⭐" if score > 1.3 else " " if score > 1.2 else " "
            print(f"  {pattern_name:35}: {score:5.2f}/12 {star}")
        
        print("\n" + "-" * 60)
        print("🏆 EN İYİ 5 PATTERN")
        print("-" * 60)
        
        sorted_results = sorted(self.results.items(), key=lambda x: x[1], reverse=True)
        for i, (name, score) in enumerate(sorted_results[:5]):
            print(f"  {i+1}. {name:35}: {score:.2f}/12")
        
        return self.results


# ============================================================
# ANA SINIF
# ============================================================

class SuperLotoPatternMaster:
    def __init__(self, excel_path="superloto.xlsx", sheet_name="s1"):
        self.excel_path = excel_path
        self.sheet_name = sheet_name
        self.df = None
        self.test_results = None
        
    def load_data(self):
        loader = DataLoader(self.excel_path, self.sheet_name)
        self.df = loader.load()
        self.get_numbers = loader.get_numbers
        print(f"✅ Veri yüklendi: {len(self.df)} çekiliş")
        if len(self.df) > 0:
            print(f"📅 Aralık: {self.df['tarih'].min().strftime('%d.%m.%Y')} - {self.df['tarih'].max().strftime('%d.%m.%Y')}")
        return self.df
    
    def run_tests(self, test_size=150):
        engine = BacktestEngine(self.df, self.get_numbers)
        self.test_results = engine.run_all_tests(test_size)
        
        best_pattern = max(self.test_results.items(), key=lambda x: x[1])
        print(f"\n🎯 EN İYİ PATTERN: {best_pattern[0]} ({best_pattern[1]:.2f}/12)")
        print(f"   Rastgele şans: 1.20/12")
        print(f"   İyileştirme: %{((best_pattern[1]-1.2)/1.2*100):.1f}")
        
        return self.test_results
    
    def get_best_pattern_predictions(self):
        """En iyi pattern ile tahmin yap"""
        if self.test_results is None:
            self.run_tests()
        
        best_pattern_name = max(self.test_results.items(), key=lambda x: x[1])[0]
        
        # Pattern adından fonksiyon adını bul
        pattern_map = {
            'Son 1 çekiliş': 'p_son_1',
            'Son 3 çekiliş': 'p_son_3',
            'Son 5 çekiliş': 'p_son_5',
            'Son 7 çekiliş': 'p_son_7',
            'Son 10 çekiliş': 'p_son_10',
            'Son 15 çekiliş': 'p_son_15',
            'Son 20 çekiliş': 'p_son_20',
            'Son 30 çekiliş': 'p_son_30',
            'Fibonacci sayıları': 'p_fibonacci',
            'Fibonacci + komşu': 'p_fibonacci_neighbor',
            'Fibonacci aralığı': 'p_fibonacci_range',
            'Sadece tek sayılar': 'p_only_odd',
            'Sadece çift sayılar': 'p_only_even',
            '3 tek + 3 çift': 'p_3odd_3even',
            '4 tek + 2 çift': 'p_4odd_2even',
            'Küçük sayılar (1-30)': 'p_small',
            'Büyük sayılar (31-60)': 'p_large',
            '3 küçük + 3 büyük': 'p_3small_3large',
            "Mod 5 (5'in katları)": 'p_mod5',
            "Mod 10 (10'un katları)": 'p_mod10',
            "Mod 3 (3'ün katları)": 'p_mod3',
            'Asal sayılar': 'p_prime',
            'Tam kareler': 'p_perfect_square',
            "2'nin kuvvetleri": 'p_power2',
            'Due numbers (en uzun süredir çıkmayan)': 'p_due',
            'Hot numbers (en çok çıkan)': 'p_hot',
            'Trend artan sayılar': 'p_trend_up',
            'Trend azalan sayılar': 'p_trend_down',
            'Son çekilişin komşuları (±1,±2)': 'p_neighbor',
            'Son çekilişin aynısı': 'p_last_draw',
            'Aralık analizi': 'p_interval',
            'Haftanın aynı günü': 'p_weekday',
            'Ayın aynı günü': 'p_month_day',
            'Çift numaralı çekilişler': 'p_even_draws',
            'Tek numaralı çekilişler': 'p_odd_draws',
        }
        
        pattern_func = pattern_map.get(best_pattern_name, 'p_hot')
        patterns = PatternMaster(self.df, self.get_numbers)
        
        try:
            predictions = getattr(patterns, pattern_func)()
        except:
            predictions = patterns.p_hot()
        
        return predictions[:12], predictions[:6]
    
    def print_report(self):
        if self.test_results is None:
            self.run_tests()
        
        best_12, best_6 = self.get_best_pattern_predictions()
        best_pattern_name = max(self.test_results.items(), key=lambda x: x[1])[0]
        best_score = max(self.test_results.values())
        
        print("\n" + "=" * 60)
        print("🎯 SÜPER LOTO PATTERN MASTER RAPORU")
        print("   35 pattern test edildi, 150 çekiliş backtest")
        print("=" * 60)
        print(f"\n📅 Son Çekiliş: {self.df['tarih'].max().strftime('%d.%m.%Y')}")
        print(f"📊 Toplam Analiz: {len(self.df)} çekiliş")
        print(f"🏆 En İyi Pattern: {best_pattern_name}")
        print(f"📈 Başarısı: {best_score:.2f}/12 (Rastgele: 1.20/12)")
        
        print("\n" + "-" * 60)
        print("🏆 EN İYİ PATTERN İLE 12 SAYI")
        print("-" * 60)
        
        for i in range(0, len(best_12), 4):
            group = best_12[i:i+4]
            print(f"  {i+1:2d}-{i+4:2d}. {' '.join(f'{n:3d}' for n in group)}")
        
        print("\n" + "-" * 60)
        print("🎯 ÖNERİLEN 6 SAYI (Bu 12'nin içinden en sık çıkanlar)")
        print("-" * 60)
        print(f"\n  🌟🌟🌟  {best_6}  🌟🌟🌟")
        
        print("\n" + "-" * 60)
        print("📊 TÜM PATTERN SONUÇLARI (Sıralı)")
        print("-" * 60)
        
        sorted_results = sorted(self.test_results.items(), key=lambda x: x[1], reverse=True)
        for i, (name, score) in enumerate(sorted_results):
            if score > 1.3:
                star = "🔥"
            elif score > 1.2:
                star = "⭐"
            else:
                star = "  "
            print(f"  {i+1:2d}. {name:35}: {score:.2f}/12 {star}")
        
        print("\n" + "-" * 60)
        print("⚠️ NOT: Bu tahminler istatistiksel analizdir.")
        print("   Kesin sonuç garantisi yoktur. Eğlence amaçlıdır.")
        print("=" * 60)
        
        return {'best_12': best_12, 'best_6': best_6}
    
    def save_results(self, best_12, best_6):
        os.makedirs('outputs', exist_ok=True)
        
        with open('outputs/pattern_master_results.json', 'w', encoding='utf-8') as f:
            json.dump({
                'best_12_numbers': best_12,
                'recommended_6_numbers': best_6,
                'best_pattern': max(self.test_results.items(), key=lambda x: x[1])[0],
                'all_pattern_results': self.test_results
            }, f, ensure_ascii=False, indent=2)
        
        with open('outputs/pattern_master_report.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("🎯 SÜPER LOTO PATTERN MASTER RAPORU\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Son Çekiliş: {self.df['tarih'].max().strftime('%d.%m.%Y')}\n")
            f.write(f"Toplam Analiz: {len(self.df)} çekiliş\n\n")
            f.write("EN İYİ PATTERN:\n")
            best = max(self.test_results.items(), key=lambda x: x[1])
            f.write(f"  {best[0]}: {best[1]:.2f}/12\n\n")
            f.write("EN GÜÇLÜ 12 SAYI:\n")
            f.write(str(best_12) + "\n\n")
            f.write("ÖNERİLEN 6 SAYI:\n")
            f.write(str(best_6) + "\n\n")
            f.write("TÜM PATTERN SONUÇLARI:\n")
            for name, score in sorted(self.test_results.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {name}: {score:.2f}/12\n")
        
        print(f"\n💾 Kaydedildi: outputs/pattern_master_results.json")
        print(f"💾 Kaydedildi: outputs/pattern_master_report.txt")


# ============================================================
# ANA ÇALIŞTIRMA
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("🚀 SÜPER LOTO PATTERN MASTER BOTU")
    print("   35 pattern test edilecek")
    print("   903 çekiliş, 150 backtest")
    print("=" * 60)
    
    bot = SuperLotoPatternMaster()
    bot.load_data()
    bot.run_tests(test_size=150)
    result = bot.print_report()
    bot.save_results(result['best_12'], result['best_6'])
    
    print("\n✅ TAMAMLANDI!")

if __name__ == "__main__":
    main()
