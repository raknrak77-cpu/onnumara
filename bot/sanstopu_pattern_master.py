#!/usr/bin/env python3
"""
ŞANS TOPU PATTERN MASTER BOTU
35+ pattern test eder, hangisi gerçekten işe yarıyor bulur
591 çekiliş, 100 backtest
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
        self.plus_column = 'no_5+1'  # Artı numara sütunu
        
    def load(self):
        self.df = pd.read_excel(self.excel_path, sheet_name=self.sheet_name, header=0)
        
        # Tarih sütununu düzenle
        if 'tarih' in self.df.columns:
            self.df['tarih'] = pd.to_datetime(self.df['tarih'], format='%d/%m/%Y', errors='coerce')
        elif 'tarih.1' in self.df.columns:
            self.df['tarih'] = pd.to_datetime(self.df['tarih.1'], format='%d/%m/%Y', errors='coerce')
        
        # Sayı sütunlarını numeric yap
        for col in self.main_columns + [self.plus_column]:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Nan'ları temizle
        self.df = self.df.dropna(subset=['tarih'] + self.main_columns + [self.plus_column], how='any')
        self.df = self.df.reset_index(drop=True)
        
        # Ek bilgiler
        self.df['weekday'] = self.df['tarih'].dt.weekday
        self.df['month'] = self.df['tarih'].dt.month
        self.df['year'] = self.df['tarih'].dt.year
        self.df['day'] = self.df['tarih'].dt.day
        
        return self.df
    
    def get_main_numbers(self, row):
        """Ana 5 sayıyı al"""
        nums = []
        for col in self.main_columns:
            if col in row.index and pd.notna(row[col]):
                try:
                    nums.append(int(row[col]))
                except:
                    pass
        return nums
    
    def get_plus_number(self, row):
        """Artı 1 sayıyı al"""
        if self.plus_column in row.index and pd.notna(row[self.plus_column]):
            try:
                return int(row[self.plus_column])
            except:
                return 0
        return 0


# ============================================================
# ANA KISIM PATTERNLERİ (1-34 arası 5 sayı)
# ============================================================

class MainPatterns:
    def __init__(self, df, get_main_func):
        self.df = df
        self.get_main = get_main_func
    
    # ===== ZAMAN TABANLI =====
    def p_main_son_n(self, n, k=10):
        recent_nums = []
        window = min(n, len(self.df))
        for idx in range(len(self.df) - window, len(self.df)):
            recent_nums.extend(self.get_main(self.df.iloc[idx]))
        counter = Counter(recent_nums)
        return [num for num, _ in counter.most_common(k)]
    
    def p_main_son_1(self): return self.p_main_son_n(1, 10)
    def p_main_son_3(self): return self.p_main_son_n(3, 10)
    def p_main_son_5(self): return self.p_main_son_n(5, 10)
    def p_main_son_7(self): return self.p_main_son_n(7, 10)
    def p_main_son_10(self): return self.p_main_son_n(10, 10)
    def p_main_son_15(self): return self.p_main_son_n(15, 10)
    def p_main_son_20(self): return self.p_main_son_n(20, 10)
    def p_main_son_30(self): return self.p_main_son_n(30, 10)
    
    # ===== FİBONACCİ =====
    def p_main_fibonacci(self, k=10):
        fib = [1, 2, 3, 5, 8, 13, 21, 34]
        return [f for f in fib if f <= 34][:k]
    
    def p_main_fibonacci_neighbor(self, k=10):
        fib = [1, 2, 3, 5, 8, 13, 21, 34]
        neighbors = set()
        for f in fib:
            neighbors.add(f)
            if f > 1: neighbors.add(f-1)
            if f < 34: neighbors.add(f+1)
        return sorted(list(neighbors))[:k]
    
    def p_main_fibonacci_range(self, k=10):
        fib = [1, 2, 3, 5, 8, 13, 21, 34]
        ranges = set()
        for i in range(len(fib)-1):
            for x in range(fib[i], fib[i+1]+1):
                if x <= 34:
                    ranges.add(x)
        return sorted(list(ranges))[:k]
    
    # ===== TEK/ÇİFT =====
    def p_main_only_odd(self, k=10):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_main(row))
        odds = [num for num in all_nums if num % 2 == 1]
        counter = Counter(odds)
        return [num for num, _ in counter.most_common(k)]
    
    def p_main_only_even(self, k=10):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_main(row))
        evens = [num for num in all_nums if num % 2 == 0]
        counter = Counter(evens)
        return [num for num, _ in counter.most_common(k)]
    
    def p_main_3odd_2even(self, k=10):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_main(row))
        odds = [num for num in all_nums if num % 2 == 1]
        evens = [num for num in all_nums if num % 2 == 0]
        odd_counter = Counter(odds)
        even_counter = Counter(evens)
        top_odds = [num for num, _ in odd_counter.most_common(6)]
        top_evens = [num for num, _ in even_counter.most_common(4)]
        return (top_odds + top_evens)[:k]
    
    def p_main_4odd_1even(self, k=10):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_main(row))
        odds = [num for num in all_nums if num % 2 == 1]
        evens = [num for num in all_nums if num % 2 == 0]
        odd_counter = Counter(odds)
        even_counter = Counter(evens)
        top_odds = [num for num, _ in odd_counter.most_common(8)]
        top_evens = [num for num, _ in even_counter.most_common(2)]
        return (top_odds + top_evens)[:k]
    
    # ===== BÜYÜK/KÜÇÜK =====
    def p_main_small(self, k=10):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_main(row))
        small = [num for num in all_nums if num <= 17]
        counter = Counter(small)
        return [num for num, _ in counter.most_common(k)]
    
    def p_main_large(self, k=10):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_main(row))
        large = [num for num in all_nums if num > 17]
        counter = Counter(large)
        return [num for num, _ in counter.most_common(k)]
    
    def p_main_3small_2large(self, k=10):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_main(row))
        small = [num for num in all_nums if num <= 17]
        large = [num for num in all_nums if num > 17]
        small_counter = Counter(small)
        large_counter = Counter(large)
        top_small = [num for num, _ in small_counter.most_common(6)]
        top_large = [num for num, _ in large_counter.most_common(4)]
        return (top_small + top_large)[:k]
    
    # ===== HOT & DUE =====
    def p_main_hot(self, k=10):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_main(row))
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]
    
    def p_main_due(self, k=10):
        """En uzun süredir çıkmayan sayılar"""
        last_seen = {num: 0 for num in range(1, 35)}
        for idx, row in self.df.iterrows():
            for num in self.get_main(row):
                last_seen[num] = idx
        last_idx = len(self.df) - 1
        due_counts = {num: last_idx - last_seen[num] for num in range(1, 35)}
        due_sorted = sorted(due_counts.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in due_sorted[:k]]
    
    # ===== TREND =====
    def p_main_trend_up(self, k=10, window=20):
        scores = {}
        for num in range(1, 35):
            recent_window = self.df.iloc[-window:]
            older_window = self.df.iloc[-window*2:-window] if len(self.df) > window*2 else self.df.iloc[:window]
            
            recent_count = sum(1 for _, row in recent_window.iterrows() if num in self.get_main(row))
            older_count = sum(1 for _, row in older_window.iterrows() if num in self.get_main(row))
            
            if older_count > 0:
                trend = recent_count - older_count
            else:
                trend = recent_count
            scores[num] = trend
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:k]]
    
    def p_main_trend_down(self, k=10, window=20):
        scores = {}
        for num in range(1, 35):
            recent_window = self.df.iloc[-window:]
            older_window = self.df.iloc[-window*2:-window] if len(self.df) > window*2 else self.df.iloc[:window]
            
            recent_count = sum(1 for _, row in recent_window.iterrows() if num in self.get_main(row))
            older_count = sum(1 for _, row in older_window.iterrows() if num in self.get_main(row))
            
            if older_count > 0:
                trend = older_count - recent_count
            else:
                trend = 0
            scores[num] = trend
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:k]]
    
    # ===== KOMŞULUK =====
    def p_main_neighbor(self, k=10):
        last_row = self.df.iloc[-1]
        last_nums = self.get_main(last_row)
        neighbors = set()
        for num in last_nums:
            for offset in [-2, -1, 0, 1, 2]:
                neighbor = num + offset
                if 1 <= neighbor <= 34:
                    neighbors.add(neighbor)
        return sorted(list(neighbors))[:k]
    
    def p_main_last_draw(self, k=10):
        last_row = self.df.iloc[-1]
        return self.get_main(last_row)[:k]
    
    # ===== ASAL SAYILAR =====
    def p_main_prime(self, k=10):
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        return [p for p in primes if p <= 34][:k]
    
    # ===== MODÜLER =====
    def p_main_mod5(self, k=10):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_main(row))
        mod5 = [num for num in all_nums if num % 5 == 0]
        counter = Counter(mod5)
        return [num for num, _ in counter.most_common(k)]
    
    def p_main_mod3(self, k=10):
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_main(row))
        mod3 = [num for num in all_nums if num % 3 == 0]
        counter = Counter(mod3)
        return [num for num, _ in counter.most_common(k)]
    
    # ===== ZAMAN (Haftanın günü) =====
    def p_main_weekday(self, k=10):
        last_weekday = self.df.iloc[-1]['weekday']
        same_weekday = self.df[self.df['weekday'] == last_weekday]
        all_nums = []
        for _, row in same_weekday.iterrows():
            all_nums.extend(self.get_main(row))
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]
    
    def p_main_month_day(self, k=10):
        last_day = self.df.iloc[-1]['day']
        same_day = self.df[self.df['day'] == last_day]
        if len(same_day) < 5:
            return self.p_main_hot(k)
        all_nums = []
        for _, row in same_day.iterrows():
            all_nums.extend(self.get_main(row))
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]
    
    # ===== TEK/ÇİFT ÇEKİLİŞLER =====
    def p_main_even_draws(self, k=10):
        even_indices = self.df.iloc[::2]
        all_nums = []
        for _, row in even_indices.iterrows():
            all_nums.extend(self.get_main(row))
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]
    
    def p_main_odd_draws(self, k=10):
        odd_indices = self.df.iloc[1::2]
        all_nums = []
        for _, row in odd_indices.iterrows():
            all_nums.extend(self.get_main(row))
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]


# ============================================================
# ARTI KISIM PATTERNLERİ (1-14 arası 1 sayı)
# ============================================================

class PlusPatterns:
    def __init__(self, df, get_plus_func):
        self.df = df
        self.get_plus = get_plus_func
    
    def p_plus_son_n(self, n, k=3):
        recent_nums = []
        window = min(n, len(self.df))
        for idx in range(len(self.df) - window, len(self.df)):
            num = self.get_plus(self.df.iloc[idx])
            if num > 0:
                recent_nums.append(num)
        counter = Counter(recent_nums)
        return [num for num, _ in counter.most_common(k)]
    
    def p_plus_son_1(self): return self.p_plus_son_n(1, 3)
    def p_plus_son_3(self): return self.p_plus_son_n(3, 3)
    def p_plus_son_5(self): return self.p_plus_son_n(5, 3)
    def p_plus_son_7(self): return self.p_plus_son_n(7, 3)
    def p_plus_son_10(self): return self.p_plus_son_n(10, 3)
    def p_plus_son_15(self): return self.p_plus_son_n(15, 3)
    def p_plus_son_20(self): return self.p_plus_son_n(20, 3)
    
    def p_plus_hot(self, k=3):
        all_nums = []
        for _, row in self.df.iterrows():
            num = self.get_plus(row)
            if num > 0:
                all_nums.append(num)
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]
    
    def p_plus_due(self, k=3):
        last_seen = {num: 0 for num in range(1, 15)}
        for idx, row in self.df.iterrows():
            num = self.get_plus(row)
            if num > 0:
                last_seen[num] = idx
        last_idx = len(self.df) - 1
        due_counts = {num: last_idx - last_seen[num] for num in range(1, 15)}
        due_sorted = sorted(due_counts.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in due_sorted[:k]]
    
    def p_plus_odd(self, k=3):
        all_nums = []
        for _, row in self.df.iterrows():
            num = self.get_plus(row)
            if num > 0 and num % 2 == 1:
                all_nums.append(num)
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]
    
    def p_plus_even(self, k=3):
        all_nums = []
        for _, row in self.df.iterrows():
            num = self.get_plus(row)
            if num > 0 and num % 2 == 0:
                all_nums.append(num)
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]
    
    def p_plus_small(self, k=3):
        all_nums = []
        for _, row in self.df.iterrows():
            num = self.get_plus(row)
            if num > 0 and num <= 7:
                all_nums.append(num)
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]
    
    def p_plus_large(self, k=3):
        all_nums = []
        for _, row in self.df.iterrows():
            num = self.get_plus(row)
            if num > 0 and num > 7:
                all_nums.append(num)
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]
    
    def p_plus_prime(self, k=3):
        primes = [2, 3, 5, 7, 11, 13]
        all_nums = []
        for _, row in self.df.iterrows():
            num = self.get_plus(row)
            if num > 0 and num in primes:
                all_nums.append(num)
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]
    
    def p_plus_fibonacci(self, k=3):
        fib = [1, 2, 3, 5, 8, 13]
        return [f for f in fib if f <= 14][:k]
    
    def p_plus_neighbor(self, k=3):
        last_num = self.get_plus(self.df.iloc[-1])
        if last_num == 0:
            return self.p_plus_hot(k)
        neighbors = set()
        for offset in [-2, -1, 0, 1, 2]:
            neighbor = last_num + offset
            if 1 <= neighbor <= 14:
                neighbors.add(neighbor)
        return sorted(list(neighbors))[:k]
    
    def p_plus_last_draw(self, k=3):
        last_num = self.get_plus(self.df.iloc[-1])
        return [last_num] if last_num > 0 else self.p_plus_hot(k)
    
    def p_plus_weekday(self, k=3):
        last_weekday = self.df.iloc[-1]['weekday']
        same_weekday = self.df[self.df['weekday'] == last_weekday]
        all_nums = []
        for _, row in same_weekday.iterrows():
            num = self.get_plus(row)
            if num > 0:
                all_nums.append(num)
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]


# ============================================================
# BACKTEST MOTORU
# ============================================================

class BacktestEngine:
    def __init__(self, df, get_main_func, get_plus_func):
        self.df = df
        self.get_main = get_main_func
        self.get_plus = get_plus_func
        self.main_patterns = MainPatterns(df, get_main_func)
        self.plus_patterns = PlusPatterns(df, get_plus_func)
        self.main_results = {}
        self.plus_results = {}
    
    def test_main_pattern(self, pattern_func, pattern_name, test_size=100):
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
            temp_patterns = MainPatterns(temp_df, self.get_main)
            
            try:
                preds = getattr(temp_patterns, pattern_func)()
            except:
                preds = []
            
            test_row = self.df.iloc[train_end]
            actual = set(self.get_main(test_row))
            correct = len(set(preds) & actual)
            scores.append(correct)
        
        avg_score = sum(scores) / len(scores) if scores else 0
        return avg_score
    
    def test_plus_pattern(self, pattern_func, pattern_name, test_size=100):
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
            temp_patterns = PlusPatterns(temp_df, self.get_plus)
            
            try:
                preds = getattr(temp_patterns, pattern_func)()
            except:
                preds = []
            
            test_row = self.df.iloc[train_end]
            actual = self.get_plus(test_row)
            if actual > 0:
                # 3 tahmin yapıyoruz, herhangi biri tutarsa +1
                correct = 1 if actual in preds else 0
                scores.append(correct)
        
        avg_score = sum(scores) / len(scores) if scores else 0
        return avg_score
    
    def run_all_tests(self, test_size=100):
        print(f"\n📊 PATTERN TESTİ BAŞLIYOR (son {test_size} çekiliş üzerinde)")
        print("=" * 60)
        
        # ANA KISIM PATTERNLERİ
        main_pattern_list = [
            ('p_main_son_1', 'Son 1 çekiliş'),
            ('p_main_son_3', 'Son 3 çekiliş'),
            ('p_main_son_5', 'Son 5 çekiliş'),
            ('p_main_son_7', 'Son 7 çekiliş'),
            ('p_main_son_10', 'Son 10 çekiliş'),
            ('p_main_son_15', 'Son 15 çekiliş'),
            ('p_main_son_20', 'Son 20 çekiliş'),
            ('p_main_son_30', 'Son 30 çekiliş'),
            ('p_main_fibonacci', 'Fibonacci sayıları'),
            ('p_main_fibonacci_neighbor', 'Fibonacci + komşu'),
            ('p_main_fibonacci_range', 'Fibonacci aralığı'),
            ('p_main_only_odd', 'Sadece tek sayılar'),
            ('p_main_only_even', 'Sadece çift sayılar'),
            ('p_main_3odd_2even', '3 tek + 2 çift'),
            ('p_main_4odd_1even', '4 tek + 1 çift'),
            ('p_main_small', 'Küçük sayılar (1-17)'),
            ('p_main_large', 'Büyük sayılar (18-34)'),
            ('p_main_3small_2large', '3 küçük + 2 büyük'),
            ('p_main_hot', 'Hot numbers (en çok çıkan)'),
            ('p_main_due', 'Due numbers (en uzun süredir çıkmayan)'),
            ('p_main_trend_up', 'Trend artan sayılar'),
            ('p_main_trend_down', 'Trend azalan sayılar'),
            ('p_main_neighbor', 'Son çekilişin komşuları'),
            ('p_main_last_draw', 'Son çekilişin aynısı'),
            ('p_main_prime', 'Asal sayılar'),
            ('p_main_mod5', "Mod 5 (5'in katları)"),
            ('p_main_mod3', "Mod 3 (3'ün katları)"),
            ('p_main_weekday', 'Haftanın aynı günü'),
            ('p_main_month_day', 'Ayın aynı günü'),
            ('p_main_even_draws', 'Çift numaralı çekilişler'),
            ('p_main_odd_draws', 'Tek numaralı çekilişler'),
        ]
        
        print("\n🎯 ANA KISIM PATTERN SONUÇLARI (5 sayı / 10 tahmin)")
        print("-" * 55)
        print(f"{'Pattern':35} {'Başarı':>10}")
        print("-" * 55)
        
        for pattern_func, pattern_name in main_pattern_list:
            score = self.test_main_pattern(pattern_func, pattern_name, test_size)
            self.main_results[pattern_name] = score
            rastgele = 10 * (5/34)  # 1.47
            improvement = ((score - rastgele) / rastgele * 100) if rastgele > 0 else 0
            star = "🔥" if improvement > 20 else "⭐" if improvement > 10 else " " if improvement > 0 else "❌"
            print(f"  {pattern_name:35}: {score:5.2f}/10 ({improvement:+.0f}%) {star}")
        
        print("\n" + "-" * 55)
        
        # ARTI KISIM PATTERNLERİ
        plus_pattern_list = [
            ('p_plus_son_1', 'Son 1 çekiliş'),
            ('p_plus_son_3', 'Son 3 çekiliş'),
            ('p_plus_son_5', 'Son 5 çekiliş'),
            ('p_plus_son_7', 'Son 7 çekiliş'),
            ('p_plus_son_10', 'Son 10 çekiliş'),
            ('p_plus_son_15', 'Son 15 çekiliş'),
            ('p_plus_son_20', 'Son 20 çekiliş'),
            ('p_plus_hot', 'Hot numbers'),
            ('p_plus_due', 'Due numbers'),
            ('p_plus_odd', 'Tek sayılar'),
            ('p_plus_even', 'Çift sayılar'),
            ('p_plus_small', 'Küçük sayılar (1-7)'),
            ('p_plus_large', 'Büyük sayılar (8-14)'),
            ('p_plus_prime', 'Asal sayılar'),
            ('p_plus_fibonacci', 'Fibonacci sayıları'),
            ('p_plus_neighbor', 'Son çekilişin komşuları'),
            ('p_plus_last_draw', 'Son çekilişin aynısı'),
            ('p_plus_weekday', 'Haftanın aynı günü'),
        ]
        
        print("\n🎯 ARTI KISIM PATTERN SONUÇLARI (1 sayı / 3 tahmin)")
        print("-" * 55)
        print(f"{'Pattern':35} {'Başarı':>10}")
        print("-" * 55)
        
        for pattern_func, pattern_name in plus_pattern_list:
            score = self.test_plus_pattern(pattern_func, pattern_name, test_size)
            self.plus_results[pattern_name] = score
            rastgele = 3 * (1/14)  # 0.214
            improvement = ((score - rastgele) / rastgele * 100) if rastgele > 0 else 0
            star = "🔥" if improvement > 20 else "⭐" if improvement > 10 else " " if improvement > 0 else "❌"
            print(f"  {pattern_name:35}: {score:5.2f}/3 ({improvement:+.0f}%) {star}")
        
        print("\n" + "=" * 60)
        
        # EN İYİLER
        best_main = max(self.main_results.items(), key=lambda x: x[1])
        best_plus = max(self.plus_results.items(), key=lambda x: x[1])
        
        rastgele_main = 10 * (5/34)
        rastgele_plus = 3 * (1/14)
        
        print(f"\n🏆 EN İYİ ANA PATTERN: {best_main[0]}")
        print(f"   Başarı: {best_main[1]:.2f}/10 (Rastgele: {rastgele_main:.2f}/10)")
        print(f"   İyileştirme: %{((best_main[1]-rastgele_main)/rastgele_main*100):.1f}")
        
        print(f"\n🏆 EN İYİ ARTI PATTERN: {best_plus[0]}")
        print(f"   Başarı: {best_plus[1]:.2f}/3 (Rastgele: {rastgele_plus:.2f}/3)")
        print(f"   İyileştirme: %{((best_plus[1]-rastgele_plus)/rastgele_plus*100):.1f}")
        
        return self.main_results, self.plus_results


# ============================================================
# ANA SINIF
# ============================================================

class SansTopuPatternMaster:
    def __init__(self, excel_path="sanstopu.xlsx", sheet_name="s1"):
        self.excel_path = excel_path
        self.sheet_name = sheet_name
        self.df = None
        self.main_results = None
        self.plus_results = None
        
    def load_data(self):
        loader = DataLoader(self.excel_path, self.sheet_name)
        self.df = loader.load()
        self.get_main = loader.get_main_numbers
        self.get_plus = loader.get_plus_number
        print(f"✅ Veri yüklendi: {len(self.df)} çekiliş")
        if len(self.df) > 0:
            print(f"📅 Aralık: {self.df['tarih'].min().strftime('%d.%m.%Y')} - {self.df['tarih'].max().strftime('%d.%m.%Y')}")
        return self.df
    
    def run_tests(self, test_size=100):
        engine = BacktestEngine(self.df, self.get_main, self.get_plus)
        self.main_results, self.plus_results = engine.run_all_tests(test_size)
        return self.main_results, self.plus_results
    
    def get_best_main_predictions(self, k=10):
        """En iyi pattern ile ana tahmin"""
        if self.main_results is None:
            self.run_tests()
        
        best_name = max(self.main_results.items(), key=lambda x: x[1])[0]
        
        pattern_map = {
            'Son 1 çekiliş': 'p_main_son_1',
            'Son 3 çekiliş': 'p_main_son_3',
            'Son 5 çekiliş': 'p_main_son_5',
            'Son 7 çekiliş': 'p_main_son_7',
            'Son 10 çekiliş': 'p_main_son_10',
            'Son 15 çekiliş': 'p_main_son_15',
            'Son 20 çekiliş': 'p_main_son_20',
            'Son 30 çekiliş': 'p_main_son_30',
            'Fibonacci sayıları': 'p_main_fibonacci',
            'Fibonacci + komşu': 'p_main_fibonacci_neighbor',
            'Fibonacci aralığı': 'p_main_fibonacci_range',
            'Sadece tek sayılar': 'p_main_only_odd',
            'Sadece çift sayılar': 'p_main_only_even',
            '3 tek + 2 çift': 'p_main_3odd_2even',
            '4 tek + 1 çift': 'p_main_4odd_1even',
            'Küçük sayılar (1-17)': 'p_main_small',
            'Büyük sayılar (18-34)': 'p_main_large',
            '3 küçük + 2 büyük': 'p_main_3small_2large',
            'Hot numbers (en çok çıkan)': 'p_main_hot',
            'Due numbers (en uzun süredir çıkmayan)': 'p_main_due',
            'Trend artan sayılar': 'p_main_trend_up',
            'Trend azalan sayılar': 'p_main_trend_down',
            'Son çekilişin komşuları': 'p_main_neighbor',
            'Son çekilişin aynısı': 'p_main_last_draw',
            'Asal sayılar': 'p_main_prime',
            "Mod 5 (5'in katları)": 'p_main_mod5',
            "Mod 3 (3'ün katları)": 'p_main_mod3',
            'Haftanın aynı günü': 'p_main_weekday',
            'Ayın aynı günü': 'p_main_month_day',
            'Çift numaralı çekilişler': 'p_main_even_draws',
            'Tek numaralı çekilişler': 'p_main_odd_draws',
        }
        
        pattern_func = pattern_map.get(best_name, 'p_main_hot')
        patterns = MainPatterns(self.df, self.get_main)
        
        try:
            predictions = getattr(patterns, pattern_func)()
        except:
            predictions = patterns.p_main_hot()
        
        return predictions[:k]
    
    def get_best_plus_predictions(self, k=3):
        """En iyi pattern ile artı tahmin"""
        if self.plus_results is None:
            self.run_tests()
        
        best_name = max(self.plus_results.items(), key=lambda x: x[1])[0]
        
        pattern_map = {
            'Son 1 çekiliş': 'p_plus_son_1',
            'Son 3 çekiliş': 'p_plus_son_3',
            'Son 5 çekiliş': 'p_plus_son_5',
            'Son 7 çekiliş': 'p_plus_son_7',
            'Son 10 çekiliş': 'p_plus_son_10',
            'Son 15 çekiliş': 'p_plus_son_15',
            'Son 20 çekiliş': 'p_plus_son_20',
            'Hot numbers': 'p_plus_hot',
            'Due numbers': 'p_plus_due',
            'Tek sayılar': 'p_plus_odd',
            'Çift sayılar': 'p_plus_even',
            'Küçük sayılar (1-7)': 'p_plus_small',
            'Büyük sayılar (8-14)': 'p_plus_large',
            'Asal sayılar': 'p_plus_prime',
            'Fibonacci sayıları': 'p_plus_fibonacci',
            'Son çekilişin komşuları': 'p_plus_neighbor',
            'Son çekilişin aynısı': 'p_plus_last_draw',
            'Haftanın aynı günü': 'p_plus_weekday',
        }
        
        pattern_func = pattern_map.get(best_name, 'p_plus_hot')
        patterns = PlusPatterns(self.df, self.get_plus)
        
        try:
            predictions = getattr(patterns, pattern_func)()
        except:
            predictions = patterns.p_plus_hot()
        
        return predictions[:k]
    
    def print_report(self):
        if self.main_results is None:
            self.run_tests()
        
        best_main_10 = self.get_best_main_predictions(10)
        best_plus_3 = self.get_best_plus_predictions(3)
        
        best_main_name = max(self.main_results.items(), key=lambda x: x[1])[0]
        best_main_score = max(self.main_results.values())
        best_plus_name = max(self.plus_results.items(), key=lambda x: x[1])[0]
        best_plus_score = max(self.plus_results.values())
        
        rastgele_main = 10 * (5/34)
        rastgele_plus = 3 * (1/14)
        
        print("\n" + "=" * 60)
        print("🎯 ŞANS TOPU PATTERN MASTER RAPORU")
        print("=" * 60)
        print(f"\n📅 Son Çekiliş: {self.df['tarih'].max().strftime('%d.%m.%Y')}")
        print(f"📊 Toplam Analiz: {len(self.df)} çekiliş")
        
        print(f"\n🏆 EN İYİ ANA PATTERN: {best_main_name}")
        print(f"   Başarı: {best_main_score:.2f}/10 (Rastgele: {rastgele_main:.2f}/10)")
        print(f"   İyileştirme: %{((best_main_score-rastgele_main)/rastgele_main*100):.1f}")
        
        print(f"\n🏆 EN İYİ ARTI PATTERN: {best_plus_name}")
        print(f"   Başarı: {best_plus_score:.2f}/3 (Rastgele: {rastgele_plus:.2f}/3)")
        print(f"   İyileştirme: %{((best_plus_score-rastgele_plus)/rastgele_plus*100):.1f}")
        
        print("\n" + "-" * 60)
        print("🎯 EN İYİ PATTERN İLE 10 ANA SAYI")
        print("-" * 60)
        
        for i in range(0, len(best_main_10), 5):
            group = best_main_10[i:i+5]
            print(f"  {i+1:2d}-{i+5:2d}. {' '.join(f'{n:3d}' for n in group)}")
        
        print("\n" + "-" * 60)
        print("🎯 EN İYİ PATTERN İLE 3 ARTI SAYI")
        print("-" * 60)
        print(f"\n  🌟🌟🌟  {best_plus_3}  🌟🌟🌟")
        
        # En iyi 5 ana pattern
        print("\n" + "-" * 60)
        print("🏆 ANA KISIM - EN İYİ 5 PATTERN")
        print("-" * 60)
        
        sorted_main = sorted(self.main_results.items(), key=lambda x: x[1], reverse=True)
        for i, (name, score) in enumerate(sorted_main[:5]):
            improvement = ((score - rastgele_main) / rastgele_main * 100) if rastgele_main > 0 else 0
            print(f"  {i+1}. {name:30}: {score:.2f}/10 (%{improvement:+.0f})")
        
        # En iyi 5 artı pattern
        print("\n" + "-" * 60)
        print("🏆 ARTI KISIM - EN İYİ 5 PATTERN")
        print("-" * 60)
        
        sorted_plus = sorted(self.plus_results.items(), key=lambda x: x[1], reverse=True)
        for i, (name, score) in enumerate(sorted_plus[:5]):
            improvement = ((score - rastgele_plus) / rastgele_plus * 100) if rastgele_plus > 0 else 0
            print(f"  {i+1}. {name:30}: {score:.2f}/3 (%{improvement:+.0f})")
        
        print("\n" + "-" * 60)
        print("⚠️ NOT: Bu tahminler istatistiksel analizdir.")
        print("   Kesin sonuç garantisi yoktur. Eğlence amaçlıdır.")
        print("=" * 60)
        
        return {'best_main_10': best_main_10, 'best_plus_3': best_plus_3}
    
    def save_results(self, best_main_10, best_plus_3):
        os.makedirs('outputs', exist_ok=True)
        
        if self.main_results:
            best_main = max(self.main_results.items(), key=lambda x: x[1])
            best_plus = max(self.plus_results.items(), key=lambda x: x[1])
        else:
            best_main = ("Bilinmiyor", 0)
            best_plus = ("Bilinmiyor", 0)
        
        with open('outputs/sanstopu_pattern_master_results.json', 'w', encoding='utf-8') as f:
            json.dump({
                'best_main_10_numbers': best_main_10,
                'best_plus_3_numbers': best_plus_3,
                'best_main_pattern': best_main[0],
                'best_main_score': best_main[1],
                'best_plus_pattern': best_plus[0],
                'best_plus_score': best_plus[1],
                'all_main_pattern_results': self.main_results,
                'all_plus_pattern_results': self.plus_results
            }, f, ensure_ascii=False, indent=2)
        
        with open('outputs/sanstopu_pattern_master_report.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("🎯 ŞANS TOPU PATTERN MASTER RAPORU\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Son Çekiliş: {self.df['tarih'].max().strftime('%d.%m.%Y')}\n")
            f.write(f"Toplam Analiz: {len(self.df)} çekiliş\n\n")
            f.write("EN İYİ ANA PATTERN:\n")
            f.write(f"  {best_main[0]}: {best_main[1]:.2f}/10\n\n")
            f.write("EN İYİ ARTI PATTERN:\n")
            f.write(f"  {best_plus[0]}: {best_plus[1]:.2f}/3\n\n")
            f.write("ÖNERİLEN 10 ANA SAYI:\n")
            f.write(str(best_main_10) + "\n\n")
            f.write("ÖNERİLEN 3 ARTI SAYI:\n")
            f.write(str(best_plus_3) + "\n\n")
        
        print(f"\n💾 Kaydedildi: outputs/sanstopu_pattern_master_results.json")
        print(f"💾 Kaydedildi: outputs/sanstopu_pattern_master_report.txt")


# ============================================================
# ANA ÇALIŞTIRMA
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("🚀 ŞANS TOPU PATTERN MASTER BOTU")
    print("   35+ pattern test edilecek")
    print("   591 çekiliş, 100 backtest")
    print("=" * 60)
    
    bot = SansTopuPatternMaster()
    bot.load_data()
    bot.run_tests(test_size=100)
    result = bot.print_report()
    bot.save_results(result['best_main_10'], result['best_plus_3'])
    
    print("\n✅ TAMAMLANDI!")

if __name__ == "__main__":
    main()
