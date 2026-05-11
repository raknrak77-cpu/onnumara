#!/usr/bin/env python3
"""
ON NUMARA PATTERN MASTER BOTU
35+ pattern test eder, hangisi gerçekten işe yarıyor bulur
Backtest: son 100 çekiliş
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
    def __init__(self, excel_path="onnumara_2020.xlsx", sheet_name="s1"):
        self.excel_path = excel_path
        self.sheet_name = sheet_name
        self.df = None
        self.number_columns = [f'no_{i}' for i in range(1, 23)]  # 22 sayı
        
    def load(self):
        self.df = pd.read_excel(self.excel_path, sheet_name=self.sheet_name, header=0)
        
        # Tarih sütununu bul
        if 'tarih' in self.df.columns:
            self.df['tarih'] = pd.to_datetime(self.df['tarih'], format='%d/%m/%Y', errors='coerce')
        elif 'Tarih' in self.df.columns:
            self.df['tarih'] = pd.to_datetime(self.df['Tarih'], format='%d/%m/%Y', errors='coerce')
        
        # Sayı sütunlarını numeric yap
        for col in self.number_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Temizlik
        all_cols = ['tarih'] + self.number_columns
        existing_cols = [c for c in all_cols if c in self.df.columns]
        self.df = self.df.dropna(subset=existing_cols, how='any')
        self.df = self.df.reset_index(drop=True)
        
        # Ek bilgiler
        self.df['weekday'] = self.df['tarih'].dt.weekday
        self.df['month'] = self.df['tarih'].dt.month
        self.df['year'] = self.df['tarih'].dt.year
        self.df['day'] = self.df['tarih'].dt.day
        
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
# PATTERNLER (1-80 arası 22 sayı)
# ============================================================

class OnNumaraPatterns:
    def __init__(self, df, get_numbers_func):
        self.df = df
        self.get_numbers = get_numbers_func
    
    # ===== ZAMAN TABANLI =====
    def p_son_n(self, n, k=22):
        recent_nums = []
        window = min(n, len(self.df))
        for idx in range(len(self.df) - window, len(self.df)):
            recent_nums.extend(self.get_numbers(self.df.iloc[idx]))
        counter = Counter(recent_nums)
        return [num for num, _ in counter.most_common(k)]
    
    def p_son_1(self): return self.p_son_n(1, 22)
    def p_son_3(self): return self.p_son_n(3, 22)
    def p_son_5(self): return self.p_son_n(5, 22)
    def p_son_7(self): return self.p_son_n(7, 22)
    def p_son_10(self): return self.p_son_n(10, 22)
    def p_son_15(self): return self.p_son_n(15, 22)
    def p_son_20(self): return self.p_son_n(20, 22)
    def p_son_30(self): return self.p_son_n(30, 22)
    def p_son_50(self): return self.p_son_n(50, 22)
    
    # ===== HOT & DUE =====
    def p_hot(self, k=22):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]
    
    def p_due(self, k=22):
        last_seen = {num: 0 for num in range(1, 81)}
        for idx, row in self.df.iterrows():
            for num in self.get_numbers(row):
                last_seen[num] = idx
        last_idx = len(self.df) - 1
        due_counts = {num: last_idx - last_seen[num] for num in range(1, 81)}
        due_sorted = sorted(due_counts.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in due_sorted[:k]]
    
    # ===== TEK/ÇİFT =====
    def p_only_odd(self, k=22):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        odds = [num for num in all_nums if num % 2 == 1]
        counter = Counter(odds)
        return [num for num, _ in counter.most_common(k)]
    
    def p_only_even(self, k=22):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        evens = [num for num in all_nums if num % 2 == 0]
        counter = Counter(evens)
        return [num for num, _ in counter.most_common(k)]
    
    def p_11odd_11even(self, k=22):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        odds = [num for num in all_nums if num % 2 == 1]
        evens = [num for num in all_nums if num % 2 == 0]
        odd_counter = Counter(odds)
        even_counter = Counter(evens)
        top_odds = [num for num, _ in odd_counter.most_common(11)]
        top_evens = [num for num, _ in even_counter.most_common(11)]
        return (top_odds + top_evens)[:k]
    
    # ===== BÜYÜK/KÜÇÜK =====
    def p_small(self, k=22):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        small = [num for num in all_nums if num <= 40]
        counter = Counter(small)
        return [num for num, _ in counter.most_common(k)]
    
    def p_large(self, k=22):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        large = [num for num in all_nums if num > 40]
        counter = Counter(large)
        return [num for num, _ in counter.most_common(k)]
    
    def p_11small_11large(self, k=22):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        small = [num for num in all_nums if num <= 40]
        large = [num for num in all_nums if num > 40]
        small_counter = Counter(small)
        large_counter = Counter(large)
        top_small = [num for num, _ in small_counter.most_common(11)]
        top_large = [num for num, _ in large_counter.most_common(11)]
        return (top_small + top_large)[:k]
    
    # ===== ÇEYREK BÖLGE =====
    def p_quarter1(self, k=22):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        q1 = [num for num in all_nums if 1 <= num <= 20]
        counter = Counter(q1)
        return [num for num, _ in counter.most_common(k)]
    
    def p_quarter2(self, k=22):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        q2 = [num for num in all_nums if 21 <= num <= 40]
        counter = Counter(q2)
        return [num for num, _ in counter.most_common(k)]
    
    def p_quarter3(self, k=22):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        q3 = [num for num in all_nums if 41 <= num <= 60]
        counter = Counter(q3)
        return [num for num, _ in counter.most_common(k)]
    
    def p_quarter4(self, k=22):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        q4 = [num for num in all_nums if 61 <= num <= 80]
        counter = Counter(q4)
        return [num for num, _ in counter.most_common(k)]
    
    # ===== FİBONACCİ =====
    def p_fibonacci(self, k=22):
        fib = [1, 2, 3, 5, 8, 13, 21, 34, 55]
        return [f for f in fib if f <= 80][:k]
    
    def p_fibonacci_neighbor(self, k=22):
        fib = [1, 2, 3, 5, 8, 13, 21, 34, 55]
        neighbors = set()
        for f in fib:
            neighbors.add(f)
            if f > 1: neighbors.add(f-1)
            if f < 80: neighbors.add(f+1)
        return sorted(list(neighbors))[:k]
    
    def p_fibonacci_range(self, k=22):
        fib = [1, 2, 3, 5, 8, 13, 21, 34, 55]
        ranges = set()
        for i in range(len(fib)-1):
            for x in range(fib[i], fib[i+1]+1):
                if x <= 80:
                    ranges.add(x)
        return sorted(list(ranges))[:k]
    
    # ===== ASAL SAYILAR =====
    def p_prime(self, k=22):
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79]
        return primes[:k]
    
    # ===== MODÜLER =====
    def p_mod5(self, k=22):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        mod5 = [num for num in all_nums if num % 5 == 0]
        counter = Counter(mod5)
        return [num for num, _ in counter.most_common(k)]
    
    def p_mod10(self, k=22):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        mod10 = [num for num in all_nums if num % 10 == 0]
        counter = Counter(mod10)
        return [num for num, _ in counter.most_common(k)]
    
    def p_mod3(self, k=22):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        mod3 = [num for num in all_nums if num % 3 == 0]
        counter = Counter(mod3)
        return [num for num, _ in counter.most_common(k)]
    
    # ===== TREND =====
    def p_trend_up(self, k=22, window=30):
        scores = {}
        for num in range(1, 81):
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
    
    def p_trend_down(self, k=22, window=30):
        scores = {}
        for num in range(1, 81):
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
    
    # ===== KOMŞULUK =====
    def p_neighbor(self, k=22):
        last_row = self.df.iloc[-1]
        last_nums = self.get_numbers(last_row)
        neighbors = set()
        for num in last_nums:
            for offset in [-3, -2, -1, 0, 1, 2, 3]:
                neighbor = num + offset
                if 1 <= neighbor <= 80:
                    neighbors.add(neighbor)
        return sorted(list(neighbors))[:k]
    
    def p_last_draw(self, k=22):
        last_row = self.df.iloc[-1]
        return self.get_numbers(last_row)[:k]
    
    # ===== ZAMAN (Haftanın günü / Ayın günü) =====
    def p_weekday(self, k=22):
        last_weekday = self.df.iloc[-1]['weekday']
        same_weekday = self.df[self.df['weekday'] == last_weekday]
        all_nums = []
        for _, row in same_weekday.iterrows():
            all_nums.extend(self.get_numbers(row))
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]
    
    def p_month_day(self, k=22):
        last_day = self.df.iloc[-1]['day']
        same_day = self.df[self.df['day'] == last_day]
        if len(same_day) < 5:
            return self.p_hot(k)
        all_nums = []
        for _, row in same_day.iterrows():
            all_nums.extend(self.get_numbers(row))
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]
    
    # ===== TEK/ÇİFT ÇEKİLİŞLER =====
    def p_even_draws(self, k=22):
        even_indices = self.df.iloc[::2]
        all_nums = []
        for _, row in even_indices.iterrows():
            all_nums.extend(self.get_numbers(row))
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]
    
    def p_odd_draws(self, k=22):
        odd_indices = self.df.iloc[1::2]
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
        self.patterns = OnNumaraPatterns(df, get_numbers_func)
        self.results = {}
    
    def test_pattern(self, pattern_func, pattern_name, test_size=100):
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
            temp_patterns = OnNumaraPatterns(temp_df, self.get_numbers)
            
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
    
    def run_all_tests(self, test_size=100):
        print(f"\n📊 ON NUMARA PATTERN TESTİ (son {test_size} çekiliş)")
        print("=" * 60)
        
        pattern_list = [
            ('p_son_1', 'Son 1 çekiliş'),
            ('p_son_3', 'Son 3 çekiliş'),
            ('p_son_5', 'Son 5 çekiliş'),
            ('p_son_7', 'Son 7 çekiliş'),
            ('p_son_10', 'Son 10 çekiliş'),
            ('p_son_15', 'Son 15 çekiliş'),
            ('p_son_20', 'Son 20 çekiliş'),
            ('p_son_30', 'Son 30 çekiliş'),
            ('p_son_50', 'Son 50 çekiliş'),
            ('p_hot', 'Hot numbers (en çok çıkan)'),
            ('p_due', 'Due numbers (en uzun süredir çıkmayan)'),
            ('p_only_odd', 'Sadece tek sayılar'),
            ('p_only_even', 'Sadece çift sayılar'),
            ('p_11odd_11even', '11 tek + 11 çift'),
            ('p_small', 'Küçük sayılar (1-40)'),
            ('p_large', 'Büyük sayılar (41-80)'),
            ('p_11small_11large', '11 küçük + 11 büyük'),
            ('p_quarter1', '1. çeyrek (1-20)'),
            ('p_quarter2', '2. çeyrek (21-40)'),
            ('p_quarter3', '3. çeyrek (41-60)'),
            ('p_quarter4', '4. çeyrek (61-80)'),
            ('p_fibonacci', 'Fibonacci sayıları'),
            ('p_fibonacci_neighbor', 'Fibonacci + komşu'),
            ('p_fibonacci_range', 'Fibonacci aralığı'),
            ('p_prime', 'Asal sayılar'),
            ('p_mod5', "Mod 5 (5'in katları)"),
            ('p_mod10', "Mod 10 (10'un katları)"),
            ('p_mod3', "Mod 3 (3'ün katları)"),
            ('p_trend_up', 'Trend artan sayılar'),
            ('p_trend_down', 'Trend azalan sayılar'),
            ('p_neighbor', 'Son çekilişin komşuları (±3)'),
            ('p_last_draw', 'Son çekilişin aynısı'),
            ('p_weekday', 'Haftanın aynı günü'),
            ('p_month_day', 'Ayın aynı günü'),
            ('p_even_draws', 'Çift numaralı çekilişler'),
            ('p_odd_draws', 'Tek numaralı çekilişler'),
        ]
        
        print("\n🎯 PATTERN SONUÇLARI (22 tahmin / çekiliş)")
        print("-" * 55)
        print(f"{'Pattern':35} {'Başarı':>10} {'vs Rastgele':>12}")
        print("-" * 55)
        
        rastgele = 22 * (22/80)  # 22 sayı çekiliyor, 80 havuz: 6.05
        print(f"  {'RASTGELE':35}: {rastgele:5.2f}/22 (referans)")
        print("-" * 55)
        
        for pattern_func, pattern_name in pattern_list:
            score = self.test_pattern(pattern_func, pattern_name, test_size)
            self.results[pattern_name] = score
            improvement = ((score - rastgele) / rastgele * 100) if rastgele > 0 else 0
            star = "🔥" if improvement > 15 else "⭐" if improvement > 5 else " " if improvement > -5 else "❌"
            print(f"  {pattern_name:35}: {score:5.2f}/22 (%{improvement:+5.1f}) {star}")
        
        print("\n" + "=" * 60)
        print("🏆 EN İYİ 10 PATTERN")
        print("-" * 60)
        
        sorted_results = sorted(self.results.items(), key=lambda x: x[1], reverse=True)
        for i, (name, score) in enumerate(sorted_results[:10]):
            improvement = ((score - rastgele) / rastgele * 100) if rastgele > 0 else 0
            print(f"  {i+1:2d}. {name:35}: {score:.2f}/22 (%{improvement:+.1f})")
        
        return self.results


# ============================================================
# ANA SINIF
# ============================================================

class OnNumaraPatternMaster:
    def __init__(self, excel_path="onnumara_2020.xlsx", sheet_name="s1"):
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
    
    def run_tests(self, test_size=100):
        engine = BacktestEngine(self.df, self.get_numbers)
        self.test_results = engine.run_all_tests(test_size)
        
        best_pattern = max(self.test_results.items(), key=lambda x: x[1])
        rastgele = 22 * (22/80)
        print(f"\n🎯 EN İYİ PATTERN: {best_pattern[0]} ({best_pattern[1]:.2f}/22)")
        print(f"   Rastgele şans: {rastgele:.2f}/22")
        print(f"   İyileştirme: %{((best_pattern[1]-rastgele)/rastgele*100):.1f}")
        
        return self.test_results
    
    def get_best_pattern_predictions(self):
        """En iyi pattern ile tahmin yap"""
        if self.test_results is None:
            self.run_tests()
        
        best_pattern_name = max(self.test_results.items(), key=lambda x: x[1])[0]
        
        pattern_map = {
            'Son 1 çekiliş': 'p_son_1',
            'Son 3 çekiliş': 'p_son_3',
            'Son 5 çekiliş': 'p_son_5',
            'Son 7 çekiliş': 'p_son_7',
            'Son 10 çekiliş': 'p_son_10',
            'Son 15 çekiliş': 'p_son_15',
            'Son 20 çekiliş': 'p_son_20',
            'Son 30 çekiliş': 'p_son_30',
            'Son 50 çekiliş': 'p_son_50',
            'Hot numbers (en çok çıkan)': 'p_hot',
            'Due numbers (en uzun süredir çıkmayan)': 'p_due',
            'Sadece tek sayılar': 'p_only_odd',
            'Sadece çift sayılar': 'p_only_even',
            '11 tek + 11 çift': 'p_11odd_11even',
            'Küçük sayılar (1-40)': 'p_small',
            'Büyük sayılar (41-80)': 'p_large',
            '11 küçük + 11 büyük': 'p_11small_11large',
            '1. çeyrek (1-20)': 'p_quarter1',
            '2. çeyrek (21-40)': 'p_quarter2',
            '3. çeyrek (41-60)': 'p_quarter3',
            '4. çeyrek (61-80)': 'p_quarter4',
            'Fibonacci sayıları': 'p_fibonacci',
            'Fibonacci + komşu': 'p_fibonacci_neighbor',
            'Fibonacci aralığı': 'p_fibonacci_range',
            'Asal sayılar': 'p_prime',
            "Mod 5 (5'in katları)": 'p_mod5',
            "Mod 10 (10'un katları)": 'p_mod10',
            "Mod 3 (3'ün katları)": 'p_mod3',
            'Trend artan sayılar': 'p_trend_up',
            'Trend azalan sayılar': 'p_trend_down',
            'Son çekilişin komşuları (±3)': 'p_neighbor',
            'Son çekilişin aynısı': 'p_last_draw',
            'Haftanın aynı günü': 'p_weekday',
            'Ayın aynı günü': 'p_month_day',
            'Çift numaralı çekilişler': 'p_even_draws',
            'Tek numaralı çekilişler': 'p_odd_draws',
        }
        
        pattern_func = pattern_map.get(best_pattern_name, 'p_hot')
        patterns = OnNumaraPatterns(self.df, self.get_numbers)
        
        try:
            predictions = getattr(patterns, pattern_func)()
        except:
            predictions = patterns.p_hot()
        
        return predictions[:22]
    
    def print_report(self):
        if self.test_results is None:
            self.run_tests()
        
        best_22 = self.get_best_pattern_predictions()
        best_pattern_name = max(self.test_results.items(), key=lambda x: x[1])[0]
        best_score = max(self.test_results.values())
        rastgele = 22 * (22/80)
        
        print("\n" + "=" * 60)
        print("🎯 ON NUMARA PATTERN MASTER RAPORU")
        print("=" * 60)
        print(f"\n📅 Son Çekiliş: {self.df['tarih'].max().strftime('%d.%m.%Y')}")
        print(f"📊 Toplam Analiz: {len(self.df)} çekiliş")
        print(f"🏆 En İyi Pattern: {best_pattern_name}")
        print(f"📈 Başarısı: {best_score:.2f}/22 (Rastgele: {rastgele:.2f}/22)")
        print(f"📈 İyileştirme: %{((best_score-rastgele)/rastgele*100):.1f}")
        
        print("\n" + "-" * 60)
        print("🏆 EN İYİ PATTERN İLE 22 SAYI")
        print("-" * 60)
        
        for i in range(0, len(best_22), 11):
            group = best_22[i:i+11]
            print(f"  {i+1:2d}-{i+11:2d}. {' '.join(f'{n:3d}' for n in group)}")
        
        print("\n" + "-" * 60)
        print("📊 TÜM PATTERN SONUÇLARI (Sıralı)")
        print("-" * 60)
        
        sorted_results = sorted(self.test_results.items(), key=lambda x: x[1], reverse=True)
        for i, (name, score) in enumerate(sorted_results[:15]):
            improvement = ((score - rastgele) / rastgele * 100) if rastgele > 0 else 0
            star = "🔥" if improvement > 15 else "⭐" if improvement > 5 else " "
            print(f"  {i+1:2d}. {name:35}: {score:.2f}/22 (%{improvement:+.1f}) {star}")
        
        print("\n" + "-" * 60)
        print("⚠️ NOT: Bu tahminler istatistiksel analizdir.")
        print("   Kesin sonuç garantisi yoktur. Eğlence amaçlıdır.")
        print("=" * 60)
        
        return {'best_22': best_22}
    
    def save_results(self, best_22):
        os.makedirs('outputs', exist_ok=True)
        
        best_pattern = max(self.test_results.items(), key=lambda x: x[1]) if self.test_results else ("Bilinmiyor", 0)
        
        with open('outputs/onnumara_pattern_master_results.json', 'w', encoding='utf-8') as f:
            json.dump({
                'best_22_numbers': best_22,
                'best_pattern': best_pattern[0],
                'best_score': best_pattern[1],
                'all_pattern_results': self.test_results
            }, f, ensure_ascii=False, indent=2)
        
        with open('outputs/onnumara_pattern_master_report.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("🎯 ON NUMARA PATTERN MASTER RAPORU\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Son Çekiliş: {self.df['tarih'].max().strftime('%d.%m.%Y')}\n")
            f.write(f"Toplam Analiz: {len(self.df)} çekiliş\n\n")
            f.write("EN İYİ PATTERN:\n")
            f.write(f"  {best_pattern[0]}: {best_pattern[1]:.2f}/22\n\n")
            f.write("EN GÜÇLÜ 22 SAYI:\n")
            f.write(str(best_22) + "\n\n")
            f.write("TÜM PATTERN SONUÇLARI:\n")
            for name, score in sorted(self.test_results.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {name}: {score:.2f}/22\n")
        
        print(f"\n💾 Kaydedildi: outputs/onnumara_pattern_master_results.json")
        print(f"💾 Kaydedildi: outputs/onnumara_pattern_master_report.txt")


# ============================================================
# ANA ÇALIŞTIRMA
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("🚀 ON NUMARA PATTERN MASTER BOTU")
    print("   35+ pattern test edilecek")
    print("   Backtest: son 100 çekiliş")
    print("=" * 60)
    
    bot = OnNumaraPatternMaster()
    bot.load_data()
    bot.run_tests(test_size=100)
    result = bot.print_report()
    bot.save_results(result['best_22'])
    
    print("\n✅ TAMAMLANDI!")

if __name__ == "__main__":
    main()
