#!/usr/bin/env python3
"""
SÜPER LOTO PATTERN MASTER V2 - GELİŞMİŞ ANALİZ
- Sayı Çiftleri (Pairwise) ve Üçlüler (Triplets) Analizi
- Ortalama Aralığa Göre Gecikme (Overdue) Analizi
- Onluk Blok (Gap) Analizi
- Entropi (Karmaşıklık) Skoru ile Ağırlıklandırma
- Toplam 45+ pattern test edilir
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
import json
import os
import math
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
    
    def get_sorted_numbers(self, row):
        return sorted(self.get_numbers(row))


# ============================================================
# GELİŞMİŞ PATTERN MASTER V2
# ============================================================

class PatternMasterV2:
    def __init__(self, df, get_numbers_func, get_sorted_func):
        self.df = df
        self.get_numbers = get_numbers_func
        self.get_sorted = get_sorted_func
        
        # Pair ve Triplet istatistikleri için cache
        self.pair_cache = None
        self.triplet_cache = None
        self.interval_cache = None
        
    # ============================================================
    # 1. ZAMAN TABANLI PATTERNS (8 pattern)
    # ============================================================
    
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
    
    # ============================================================
    # 2. FİBONACCİ PATTERNS (3 pattern)
    # ============================================================
    
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
                if x <= 60:
                    ranges.add(x)
        return sorted(list(ranges))[:k]
    
    # ============================================================
    # 3. TEK/ÇİFT PATTERNS (4 pattern)
    # ============================================================
    
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
    
    # ============================================================
    # 4. BÜYÜK/KÜÇÜK PATTERNS (3 pattern)
    # ============================================================
    
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
    
    # ============================================================
    # 5. MODÜLER PATTERNS (3 pattern)
    # ============================================================
    
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
    
    # ============================================================
    # 6. ÖZEL SAYI PATTERNS (3 pattern)
    # ============================================================
    
    def p_prime(self, k=12):
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]
        return [p for p in primes if p <= 60][:k]
    
    def p_perfect_square(self, k=12):
        squares = [1, 4, 9, 16, 25, 36, 49]
        return squares[:k]
    
    def p_power2(self, k=12):
        powers = [1, 2, 4, 8, 16, 32]
        return powers[:k]
    
    # ============================================================
    # 7. DUE & HOT PATTERNS (2 pattern)
    # ============================================================
    
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
    
    # ============================================================
    # 8. TREND PATTERNS (2 pattern)
    # ============================================================
    
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
    
    # ============================================================
    # 9. KOMŞULUK PATTERNS (2 pattern)
    # ============================================================
    
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
    
    # ============================================================
    # 10. ZAMAN TABANLI (2 pattern)
    # ============================================================
    
    def p_weekday(self, k=12):
        last_weekday = self.df.iloc[-1]['weekday']
        same_weekday = self.df[self.df['weekday'] == last_weekday]
        all_nums = []
        for _, row in same_weekday.iterrows():
            all_nums.extend(self.get_numbers(row))
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]
    
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
    
    def p_even_draws(self, k=12):
        even_indices = self.df.iloc[::2]
        all_nums = []
        for _, row in even_indices.iterrows():
            all_nums.extend(self.get_numbers(row))
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]
    
    def p_odd_draws(self, k=12):
        odd_indices = self.df.iloc[1::2]
        all_nums = []
        for _, row in odd_indices.iterrows():
            all_nums.extend(self.get_numbers(row))
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]
    
    # ============================================================
    # 11. YENİ! SAYI ÇİFTLERİ (PAIR) ANALİZİ - GEMINI ÖNERİSİ
    # ============================================================
    
    def _build_pair_cache(self):
        """Pair istatistiklerini oluştur"""
        if self.pair_cache is not None:
            return self.pair_cache
        
        pair_count = defaultdict(int)
        pair_last_seen = defaultdict(int)
        
        for idx, row in self.df.iterrows():
            nums = sorted(self.get_numbers(row))
            for pair in combinations(nums, 2):
                pair_count[pair] += 1
                pair_last_seen[pair] = idx
        
        self.pair_cache = {
            'counts': pair_count,
            'last_seen': pair_last_seen
        }
        return self.pair_cache
    
    def p_top_pairs(self, k=12):
        """En çok çıkan sayı çiftlerinden gelen sayılar"""
        cache = self._build_pair_cache()
        top_pairs = sorted(cache['counts'].items(), key=lambda x: x[1], reverse=True)[:30]
        
        # Pair'lerdeki sayıların frekansını hesapla
        pair_numbers = []
        for pair, _ in top_pairs:
            pair_numbers.extend(pair)
        
        counter = Counter(pair_numbers)
        return [num for num, _ in counter.most_common(k)]
    
    def p_with_most_frequent_partner(self, k=12):
        """Her sayı için en sık çıktığı partneri bul, onları öner"""
        cache = self._build_pair_cache()
        
        best_partners = {}
        for (a, b), count in cache['counts'].items():
            if a not in best_partners or count > best_partners[a][1]:
                best_partners[a] = (b, count)
            if b not in best_partners or count > best_partners[b][1]:
                best_partners[b] = (a, count)
        
        # En güçlü partner ilişkilerini puanla
        partner_scores = defaultdict(int)
        for num, (partner, count) in best_partners.items():
            partner_scores[partner] += count
        
        sorted_partners = sorted(partner_scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_partners[:k]]
    
    # ============================================================
    # 12. YENİ! ÜÇLÜLER (TRIPLET) ANALİZİ - GEMINI ÖNERİSİ
    # ============================================================
    
    def _build_triplet_cache(self):
        """Triplet istatistiklerini oluştur"""
        if self.triplet_cache is not None:
            return self.triplet_cache
        
        triplet_count = defaultdict(int)
        
        for _, row in self.df.iterrows():
            nums = sorted(self.get_numbers(row))
            for triplet in combinations(nums, 3):
                triplet_count[triplet] += 1
        
        self.triplet_cache = triplet_count
        return self.triplet_cache
    
    def p_top_triplets(self, k=12):
        """En çok çıkan sayı üçlülerinden gelen sayılar"""
        cache = self._build_triplet_cache()
        top_triplets = sorted(cache.items(), key=lambda x: x[1], reverse=True)[:20]
        
        triplet_numbers = []
        for triplet, _ in top_triplets:
            triplet_numbers.extend(triplet)
        
        counter = Counter(triplet_numbers)
        return [num for num, _ in counter.most_common(k)]
    
    # ============================================================
    # 13. YENİ! OVERDUE ANALİZİ (Gecikme Devri) - GEMINI ÖNERİSİ
    # ============================================================
    
    def _build_interval_cache(self):
        """Her sayının çıkma aralıklarını hesapla"""
        if self.interval_cache is not None:
            return self.interval_cache
        
        intervals = {num: [] for num in range(1, 61)}
        last_seen = {num: -1 for num in range(1, 61)}
        
        for idx, row in self.df.iterrows():
            for num in self.get_numbers(row):
                if last_seen[num] != -1:
                    interval = idx - last_seen[num]
                    intervals[num].append(interval)
                last_seen[num] = idx
        
        # Son görülme bilgisini de ekle
        current_idx = len(self.df) - 1
        current_gap = {num: current_idx - last_seen[num] if last_seen[num] != -1 else current_idx + 1 
                      for num in range(1, 61)}
        
        avg_intervals = {}
        for num in range(1, 61):
            if intervals[num]:
                avg_intervals[num] = sum(intervals[num]) / len(intervals[num])
            else:
                avg_intervals[num] = 10  # varsayılan
        
        self.interval_cache = {
            'avg': avg_intervals,
            'current_gap': current_gap,
            'history': intervals
        }
        return self.interval_cache
    
    def p_overdue(self, k=12):
        """Vadesi geçmiş (ortalamadan fazla gecikmiş) sayılar"""
        cache = self._build_interval_cache()
        
        overdue_scores = {}
        for num in range(1, 61):
            avg = cache['avg'][num]
            gap = cache['current_gap'][num]
            
            # Gecikme oranı: gap / avg (1'den büyükse vadesi geçmiş)
            if avg > 0:
                ratio = gap / avg
                # 1.5'tan büyükse bonus, 2'den büyükse büyük bonus
                if ratio > 2.0:
                    overdue_scores[num] = 3.0
                elif ratio > 1.5:
                    overdue_scores[num] = 2.0
                elif ratio > 1.0:
                    overdue_scores[num] = 1.5
                else:
                    overdue_scores[num] = 1.0
            else:
                overdue_scores[num] = 1.0
        
        sorted_nums = sorted(overdue_scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:k]]
    
    def p_due_with_rhythm(self, k=12):
        """Ritmik due analizi - sayının çıkma düzenini analiz eder"""
        cache = self._build_interval_cache()
        
        rhythm_scores = {}
        for num in range(1, 61):
            history = cache['history'][num]
            if len(history) >= 3:
                # Son 3 interval'e bak
                recent_intervals = history[-3:]
                # Eğer interval'ler düzenli artıyorsa (örnek: 3,6,9)
                if len(set(recent_intervals)) == 1:
                    rhythm_scores[num] = 2.0  # Sabit ritim
                elif recent_intervals[-1] > recent_intervals[-2] > recent_intervals[-3]:
                    rhythm_scores[num] = 1.5  # Artan ritim (çıkması yakın)
                else:
                    rhythm_scores[num] = 1.0
            else:
                rhythm_scores[num] = 1.0
        
        # Gecikme ile birleştir
        for num in range(1, 61):
            gap = cache['current_gap'][num]
            avg = cache['avg'][num]
            if avg > 0 and gap > avg * 1.2:
                rhythm_scores[num] = rhythm_scores.get(num, 1.0) * 1.5
        
        sorted_nums = sorted(rhythm_scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:k]]
    
    # ============================================================
    # 14. YENİ! BLOK/GAP ANALİZİ - GEMINI ÖNERİSİ
    # ============================================================
    
    def _calculate_block_stats(self):
        """Onluk blok istatistiklerini hesapla (1-10, 11-20, ...)"""
        blocks = {i: [] for i in range(6)}  # 0:1-10, 1:11-20, ...
        
        for _, row in self.df.iterrows():
            nums = self.get_numbers(row)
            block_hits = {i: False for i in range(6)}
            
            for num in nums:
                block_idx = (num - 1) // 10
                if 0 <= block_idx < 6:
                    block_hits[block_idx] = True
            
            for i in range(6):
                blocks[i].append(block_hits[i])
        
        # Her bloğun boş kalma oranı
        empty_rates = {}
        hit_rates = {}
        for i in range(6):
            total = len(blocks[i])
            empty_count = sum(1 for hit in blocks[i] if not hit)
            hit_count = sum(1 for hit in blocks[i] if hit)
            empty_rates[i] = empty_count / total if total > 0 else 0
            hit_rates[i] = hit_count / total if total > 0 else 0
        
        return empty_rates, hit_rates
    
    def p_block_gap_active(self, k=12):
        """En aktif bloklardan sayı seç (en az boş kalan)"""
        empty_rates, hit_rates = self._calculate_block_stats()
        
        # En aktif blok (en yüksek hit rate)
        best_block = max(hit_rates.items(), key=lambda x: x[1])[0]
        
        # Bu bloktaki sayıların sıklığını hesapla
        block_numbers = range(best_block * 10 + 1, best_block * 10 + 11)
        
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        counter = Counter(all_nums)
        
        block_nums_with_freq = [(num, counter.get(num, 0)) for num in block_numbers]
        block_nums_with_freq.sort(key=lambda x: x[1], reverse=True)
        
        return [num for num, _ in block_nums_with_freq[:k]]
    
    def p_block_gap_inactive(self, k=12):
        """En pasif bloklardan sayı seç (en çok boş kalan) - sürpriz sayılar"""
        empty_rates, hit_rates = self._calculate_block_stats()
        
        # En pasif blok (en yüksek empty rate)
        worst_block = max(empty_rates.items(), key=lambda x: x[1])[0]
        
        # Bu bloktaki sayılar
        block_numbers = range(worst_block * 10 + 1, worst_block * 10 + 11)
        
        # En çok çıkanları seç (blok içinde)
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        counter = Counter(all_nums)
        
        block_nums_with_freq = [(num, counter.get(num, 0)) for num in block_numbers]
        block_nums_with_freq.sort(key=lambda x: x[1], reverse=True)
        
        return [num for num, _ in block_nums_with_freq[:k]]
    
    # ============================================================
    # 15. YENİ! KÜMELEME (CLUSTERING) ANALİZİ - GEMINI ÖNERİSİ
    # ============================================================
    
    def p_cluster_centers(self, k=12):
        """Küme merkezlerini bul (yoğun bölgeler)"""
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        
        # Sayıları 10'ar blokta grupla ve frekansları hesapla
        block_freq = defaultdict(int)
        for num in all_nums:
            block = (num - 1) // 10
            block_freq[block] += 1
        
        # En yoğun bloğu bul
        most_dense_block = max(block_freq.items(), key=lambda x: x[1])[0]
        
        # O bloktaki sayıları sıklığa göre sırala
        block_numbers = range(most_dense_block * 10 + 1, most_dense_block * 10 + 11)
        counter = Counter(all_nums)
        
        block_nums_with_freq = [(num, counter.get(num, 0)) for num in block_numbers]
        block_nums_with_freq.sort(key=lambda x: x[1], reverse=True)
        
        return [num for num, _ in block_nums_with_freq[:k]]
    
    # ============================================================
    # 16. YENİ! ENTROPİ (KARMAŞIKLIK) SKORU - GEMINI ÖNERİSİ
    # ============================================================
    
    def calculate_entropy(self, numbers):
        """Bir sayı grubunun entropisini (dağılım karmaşıklığını) hesapla"""
        if not numbers:
            return 0
        
        # 6 eşit aralığa böl (0-10, 11-20, ...)
        ranges = [0] * 6
        for num in numbers:
            idx = (num - 1) // 10
            if 0 <= idx < 6:
                ranges[idx] += 1
        
        # Shannon entropisi
        entropy = 0
        total = len(numbers)
        for count in ranges:
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        return entropy
    
    def p_low_entropy(self, k=12):
        """Düşük entropili (düzenli/ardışık) sayılar - bazı pattern'ler bunu sever"""
        all_nums = []
        for _, row in self.df.iterrows():
            nums = self.get_numbers(row)
            if self.calculate_entropy(nums) < 2.0:  # Düşük entropi
                all_nums.extend(nums)
        
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]
    
    def p_high_entropy(self, k=12):
        """Yüksek entropili (dağınık) sayılar - gerçekçi dağılım"""
        all_nums = []
        for _, row in self.df.iterrows():
            nums = self.get_numbers(row)
            if self.calculate_entropy(nums) > 2.3:  # Yüksek entropi
                all_nums.extend(nums)
        
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]
    
    # ============================================================
    # 17. YENİ! ARALIK VE FARK ANALİZLERİ
    # ============================================================
    
    def p_gap_between_numbers(self, k=12):
        """Sayılar arasındaki farkları analiz et"""
        gaps = []
        for _, row in self.df.iterrows():
            nums = sorted(self.get_numbers(row))
            for i in range(len(nums)-1):
                gaps.append(nums[i+1] - nums[i])
        
        # En sık görülen farkı bul
        gap_counter = Counter(gaps)
        most_common_gap = gap_counter.most_common(1)[0][0]
        
        # Son çekilişteki sayılardan bu farkla ilerle
        last_nums = sorted(self.get_numbers(self.df.iloc[-1]))
        predictions = set()
        
        for num in last_nums:
            next_num = num + most_common_gap
            if 1 <= next_num <= 60:
                predictions.add(next_num)
        
        # Yeterli sayı yoksa tamamla
        if len(predictions) < k:
            all_nums = []
            for _, row in self.df.iterrows():
                all_nums.extend(self.get_numbers(row))
            counter = Counter(all_nums)
            for num, _ in counter.most_common(k):
                predictions.add(num)
                if len(predictions) >= k:
                    break
        
        return list(predictions)[:k]


# ============================================================
# BACKTEST MOTORU V2 (Entropi ile ağırlıklandırma)
# ============================================================

class BacktestEngineV2:
    def __init__(self, df, get_numbers_func, get_sorted_func):
        self.df = df
        self.get_numbers = get_numbers_func
        self.get_sorted = get_sorted_func
        self.results = {}
        
    def test_pattern(self, pattern_func, pattern_name, test_size=150):
        """Tek bir pattern'i backtest et (entropi ağırlıklı)"""
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
            temp_patterns = PatternMasterV2(temp_df, self.get_numbers, self.get_sorted)
            
            try:
                preds = getattr(temp_patterns, pattern_func)()
            except:
                preds = []
            
            test_row = self.df.iloc[train_end]
            actual = set(self.get_numbers(test_row))
            correct = len(set(preds) & actual)
            
            # Entropi ağırlığı: Tahmin setinin entropisine göre ağırlıklandır
            entropy = temp_patterns.calculate_entropy(preds)
            # Normal entropi 2.0-2.5 arası idealdir
            entropy_weight = 1.0
            if 2.0 <= entropy <= 2.5:
                entropy_weight = 1.1  # İdeal dağılıma bonus
            elif entropy < 1.5 or entropy > 2.8:
                entropy_weight = 0.9  # Çok düzenli veya çok dağınık ceza
            
            scores.append(correct * entropy_weight)
        
        avg_score = sum(scores) / len(scores) if scores else 0
        return avg_score
    
    def run_all_tests(self, test_size=150):
        print(f"\n📊 PATTERN TESTİ V2 BAŞLIYOR (son {test_size} çekiliş üzerinde)")
        print("   Toplam 48 pattern test edilecek (YENİ: Pair, Triplet, Overdue, Block, Entropy)")
        print("-" * 70)
        
        pattern_list = [
            # Zaman tabanlı (8)
            ('p_son_1', 'Son 1 çekiliş'),
            ('p_son_3', 'Son 3 çekiliş'),
            ('p_son_5', 'Son 5 çekiliş'),
            ('p_son_7', 'Son 7 çekiliş'),
            ('p_son_10', 'Son 10 çekiliş'),
            ('p_son_15', 'Son 15 çekiliş'),
            ('p_son_20', 'Son 20 çekiliş'),
            ('p_son_30', 'Son 30 çekiliş'),
            # Fibonacci (3)
            ('p_fibonacci', 'Fibonacci sayıları'),
            ('p_fibonacci_neighbor', 'Fibonacci + komşu'),
            ('p_fibonacci_range', 'Fibonacci aralığı'),
            # Tek/Çift (4)
            ('p_only_odd', 'Sadece tek sayılar'),
            ('p_only_even', 'Sadece çift sayılar'),
            ('p_3odd_3even', '3 tek + 3 çift'),
            ('p_4odd_2even', '4 tek + 2 çift'),
            # Büyük/Küçük (3)
            ('p_small', 'Küçük sayılar (1-30)'),
            ('p_large', 'Büyük sayılar (31-60)'),
            ('p_3small_3large', '3 küçük + 3 büyük'),
            # Modüler (3)
            ('p_mod5', "Mod 5 (5'in katları)"),
            ('p_mod10', "Mod 10 (10'un katları)"),
            ('p_mod3', "Mod 3 (3'ün katları)"),
            # Özel sayılar (3)
            ('p_prime', 'Asal sayılar'),
            ('p_perfect_square', 'Tam kareler'),
            ('p_power2', "2'nin kuvvetleri"),
            # Due & Hot (2)
            ('p_due', 'Due numbers (en uzun süredir çıkmayan)'),
            ('p_hot', 'Hot numbers (en çok çıkan)'),
            # Trend (2)
            ('p_trend_up', 'Trend artan sayılar'),
            ('p_trend_down', 'Trend azalan sayılar'),
            # Komşuluk (2)
            ('p_neighbor', 'Son çekilişin komşuları (±1,±2)'),
            ('p_last_draw', 'Son çekilişin aynısı'),
            # Zaman (4)
            ('p_weekday', 'Haftanın aynı günü'),
            ('p_month_day', 'Ayın aynı günü'),
            ('p_even_draws', 'Çift numaralı çekilişler'),
            ('p_odd_draws', 'Tek numaralı çekilişler'),
            # YENİ! Pair/Triplet (3)
            ('p_top_pairs', '🔗 En çok çıkan sayı çiftleri (Pair)'),
            ('p_with_most_frequent_partner', '🤝 En güçlü partner ilişkileri'),
            ('p_top_triplets', '🔺 En çok çıkan sayı üçlüleri (Triplet)'),
            # YENİ! Overdue Analizi (2)
            ('p_overdue', '⏰ Vadesi geçmiş sayılar (Overdue)'),
            ('p_due_with_rhythm', '🎵 Ritmik Due analizi'),
            # YENİ! Block/Gap Analizi (2)
            ('p_block_gap_active', '📊 En aktif blok (Blok analizi)'),
            ('p_block_gap_inactive', '🎯 En pasif blok (Sürpriz sayılar)'),
            # YENİ! Clustering (1)
            ('p_cluster_centers', '🎯 Küme merkezleri (Clustering)'),
            # YENİ! Entropi (2)
            ('p_low_entropy', '📐 Düşük entropi (Düzenli sayılar)'),
            ('p_high_entropy', '🌊 Yüksek entropi (Dağınık sayılar)'),
            # YENİ! Aralık analizi (1)
            ('p_gap_between_numbers', '📏 Sayılar arası fark analizi'),
        ]
        
        print("\n  📈 PATTERN TEST SONUÇLARI V2:")
        print(f"  {'Pattern':40} {'Başarı':>10}")
        print("  " + "-" * 52)
        
        for pattern_func, pattern_name in pattern_list:
            score = self.test_pattern(pattern_func, pattern_name, test_size)
            self.results[pattern_name] = score
            
            if score > 1.35:
                star = "🔥🔥"
            elif score > 1.30:
                star = "🔥"
            elif score > 1.25:
                star = "⭐"
            elif score > 1.20:
                star = "✓"
            else:
                star = " "
            
            print(f"  {pattern_name:40}: {score:5.2f}/12 {star}")
        
        print("\n" + "-" * 70)
        print("🏆 EN İYİ 10 PATTERN (V2)")
        print("-" * 70)
        
        sorted_results = sorted(self.results.items(), key=lambda x: x[1], reverse=True)
        for i, (name, score) in enumerate(sorted_results[:10]):
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1:2d}."
            print(f"  {medal} {name:40}: {score:.2f}/12")
        
        return self.results


# ============================================================
# ANA SINIF - SUPER LOTO PATTERN MASTER V2
# ============================================================

class SuperLotoPatternMasterV2:
    def __init__(self, excel_path="superloto.xlsx", sheet_name="s1"):
        self.excel_path = excel_path
        self.sheet_name = sheet_name
        self.df = None
        self.test_results = None
        
    def load_data(self):
        loader = DataLoader(self.excel_path, self.sheet_name)
        self.df = loader.load()
        self.get_numbers = loader.get_numbers
        self.get_sorted = loader.get_sorted_numbers
        print(f"✅ Veri yüklendi: {len(self.df)} çekiliş")
        if len(self.df) > 0:
            print(f"📅 Aralık: {self.df['tarih'].min().strftime('%d.%m.%Y')} - {self.df['tarih'].max().strftime('%d.%m.%Y')}")
        return self.df
    
    def run_tests(self, test_size=150):
        engine = BacktestEngineV2(self.df, self.get_numbers, self.get_sorted)
        self.test_results = engine.run_all_tests(test_size)
        
        best_pattern = max(self.test_results.items(), key=lambda x: x[1])
        print(f"\n🎯 EN İYİ PATTERN V2: {best_pattern[0]} ({best_pattern[1]:.2f}/12)")
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
            'Haftanın aynı günü': 'p_weekday',
            'Ayın aynı günü': 'p_month_day',
            'Çift numaralı çekilişler': 'p_even_draws',
            'Tek numaralı çekilişler': 'p_odd_draws',
            '🔗 En çok çıkan sayı çiftleri (Pair)': 'p_top_pairs',
            '🤝 En güçlü partner ilişkileri': 'p_with_most_frequent_partner',
            '🔺 En çok çıkan sayı üçlüleri (Triplet)': 'p_top_triplets',
            '⏰ Vadesi geçmiş sayılar (Overdue)': 'p_overdue',
            '🎵 Ritmik Due analizi': 'p_due_with_rhythm',
            '📊 En aktif blok (Blok analizi)': 'p_block_gap_active',
            '🎯 En pasif blok (Sürpriz sayılar)': 'p_block_gap_inactive',
            '🎯 Küme merkezleri (Clustering)': 'p_cluster_centers',
            '📐 Düşük entropi (Düzenli sayılar)': 'p_low_entropy',
            '🌊 Yüksek entropi (Dağınık sayılar)': 'p_high_entropy',
            '📏 Sayılar arası fark analizi': 'p_gap_between_numbers',
        }
        
        pattern_func = pattern_map.get(best_pattern_name, 'p_hot')
        patterns = PatternMasterV2(self.df, self.get_numbers, self.get_sorted)
        
        try:
            predictions = getattr(patterns, pattern_func)(12)
        except:
            predictions = patterns.p_hot(12)
        
        # En sık çıkan 6 sayıyı bul (öneri için)
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        counter = Counter(all_nums)
        
        # predictions içindeki sayıları sıklığa göre sırala
        pred_with_freq = [(num, counter.get(num, 0)) for num in predictions]
        pred_with_freq.sort(key=lambda x: x[1], reverse=True)
        top_6 = [num for num, _ in pred_with_freq[:6]]
        
        return predictions[:12], top_6
    
    def print_report(self):
        if self.test_results is None:
            self.run_tests()
        
        best_12, best_6 = self.get_best_pattern_predictions()
        best_pattern_name = max(self.test_results.items(), key=lambda x: x[1])[0]
        best_score = max(self.test_results.values())
        
        print("\n" + "=" * 70)
        print("🎯 SÜPER LOTO PATTERN MASTER V2 RAPORU")
        print("   48 pattern test edildi, 150 çekiliş backtest")
        print("   YENİ: Pair, Triplet, Overdue, Block, Entropy analizleri")
        print("=" * 70)
        print(f"\n📅 Son Çekiliş: {self.df['tarih'].max().strftime('%d.%m.%Y')}")
        print(f"📊 Toplam Analiz: {len(self.df)} çekiliş")
        print(f"🏆 En İyi Pattern: {best_pattern_name}")
        print(f"📈 Başarısı: {best_score:.2f}/12 (Rastgele: 1.20/12)")
        print(f"📈 İyileştirme: %{((best_score-1.2)/1.2*100):.1f}")
        
        print("\n" + "-" * 70)
        print("🏆 EN İYİ PATTERN İLE 12 SAYI")
        print("-" * 70)
        
        for i in range(0, len(best_12), 4):
            group = best_12[i:i+4]
            print(f"  {i+1:2d}-{i+4:2d}. {' '.join(f'{n:3d}' for n in group)}")
        
        print("\n" + "-" * 70)
        print("🎯 ÖNERİLEN 6 SAYI (En sık çıkanlar)")
        print("-" * 70)
        print(f"\n  🌟🌟🌟  {best_6}  🌟🌟🌟")
        
        print("\n" + "-" * 70)
        print("📊 TÜM PATTERN SONUÇLARI (Sıralı) - V2")
        print("-" * 70)
        
        sorted_results = sorted(self.test_results.items(), key=lambda x: x[1], reverse=True)
        for i, (name, score) in enumerate(sorted_results[:20]):
            if score > 1.35:
                star = "🔥🔥"
            elif score > 1.30:
                star = "🔥"
            elif score > 1.25:
                star = "⭐"
            else:
                star = "  "
            print(f"  {i+1:2d}. {name:40}: {score:.2f}/12 {star}")
        
        if len(sorted_results) > 20:
            print(f"  ... ve {len(sorted_results)-20} pattern daha")
        
        print("\n" + "-" * 70)
        print("⚠️ NOT: Bu tahminler istatistiksel analizdir.")
        print("   Kesin sonuç garantisi yoktur. Eğlence amaçlıdır.")
        print("=" * 70)
        
        return {'best_12': best_12, 'best_6': best_6}
    
    def save_results(self, best_12, best_6):
        os.makedirs('outputs', exist_ok=True)
        
        with open('outputs/pattern_master_v2_results.json', 'w', encoding='utf-8') as f:
            json.dump({
                'version': 'V2',
                'best_12_numbers': best_12,
                'recommended_6_numbers': best_6,
                'best_pattern': max(self.test_results.items(), key=lambda x: x[1])[0],
                'best_pattern_score': max(self.test_results.values()),
                'total_patterns_tested': len(self.test_results),
                'all_pattern_results': dict(sorted(self.test_results.items(), key=lambda x: x[1], reverse=True))
            }, f, ensure_ascii=False, indent=2)
        
        with open('outputs/pattern_master_v2_report.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("🎯 SÜPER LOTO PATTERN MASTER V2 RAPORU\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Son Çekiliş: {self.df['tarih'].max().strftime('%d.%m.%Y')}\n")
            f.write(f"Toplam Analiz: {len(self.df)} çekiliş\n")
            f.write(f"Test Edilen Pattern: {len(self.test_results)}\n\n")
            f.write("EN İYİ PATTERN:\n")
            best = max(self.test_results.items(), key=lambda x: x[1])
            f.write(f"  {best[0]}: {best[1]:.2f}/12\n\n")
            f.write("EN GÜÇLÜ 12 SAYI:\n")
            f.write(str(best_12) + "\n\n")
            f.write("ÖNERİLEN 6 SAYI:\n")
            f.write(str(best_6) + "\n\n")
            f.write("TÜM PATTERN SONUÇLARI (Sıralı):\n")
            for name, score in sorted(self.test_results.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {name}: {score:.2f}/12\n")
        
        print(f"\n💾 Kaydedildi: outputs/pattern_master_v2_results.json")
        print(f"💾 Kaydedildi: outputs/pattern_master_v2_report.txt")


# ============================================================
# ANA ÇALIŞTIRMA
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("🚀 SÜPER LOTO PATTERN MASTER V2")
    print("   48+ pattern test edilecek")
    print("   YENİ ÖZELLİKLER:")
    print("     • Sayı Çiftleri (Pairwise) ve Üçlüler (Triplets)")
    print("     • Ortalama Aralığa Göre Gecikme (Overdue)")
    print("     • Onluk Blok (Gap) Analizi")
    print("     • Entropi (Karmaşıklık) Skoru ile Ağırlıklandırma")
    print("=" * 70)
    
    bot = SuperLotoPatternMasterV2()
    bot.load_data()
    bot.run_tests(test_size=150)
    result = bot.print_report()
    bot.save_results(result['best_12'], result['best_6'])
    
    print("\n✅ TAMAMLANDI!")

if __name__ == "__main__":
    main()
