#!/usr/bin/env python3
"""
SÜPER LOTO PATTERN MASTER V3 - SÜPER HAVUZ
- Markov Zinciri (Sayısal Hafıza)
- Z-Score (Anomali Tespiti)
- Poisson Dağılımı (Beklenti Analizi)
- Delta Analizi (Sayılar arası farklar)
- 4 Hücreli Havuz Yapısı (Trend + Beklenti + Korelasyon + Kaos)
- ÇIKTI: 28 sayılık SÜPER HAVUZ (kombinasyon yok)
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
# PATTERN MASTER V3 - İLERİ ANALİZ
# ============================================================

class PatternMasterV3:
    def __init__(self, df, get_numbers_func):
        self.df = df
        self.get_numbers = get_numbers_func
        
        # Cache'ler
        self.markov_cache = None
        self.pair_cache = None
        
    # ============================================================
    # 1. MARKOV ZİNCİRİ (Sayısal Hafıza) - YENİ!
    # ============================================================
    
    def _build_markov_matrix(self):
        """Markov geçiş matrisi: Bir sayıdan sonra hangi sayı geliyor?"""
        if self.markov_cache is not None:
            return self.markov_cache
        
        transitions = defaultdict(lambda: defaultdict(int))
        
        for idx in range(len(self.df) - 1):
            current_nums = set(self.get_numbers(self.df.iloc[idx]))
            next_nums = set(self.get_numbers(self.df.iloc[idx + 1]))
            
            for current in current_nums:
                for next_num in next_nums:
                    transitions[current][next_num] += 1
        
        # Normalize et
        markov_prob = {}
        for current, next_dict in transitions.items():
            total = sum(next_dict.values())
            markov_prob[current] = {num: count/total for num, count in next_dict.items()}
        
        self.markov_cache = markov_prob
        return markov_prob
    
    def get_markov_boost(self):
        """Son çekilişteki sayılardan sonra gelme olasılığı yüksek olan sayılar"""
        markov = self._build_markov_matrix()
        last_nums = set(self.get_numbers(self.df.iloc[-1]))
        
        boost_scores = defaultdict(float)
        for num in last_nums:
            if num in markov:
                for next_num, prob in markov[num].items():
                    boost_scores[next_num] += prob
        
        return dict(boost_scores)
    
    # ============================================================
    # 2. Z-SCORE (Anomali Tespiti) - YENİ!
    # ============================================================
    
    def calculate_z_scores(self):
        """Her sayının çıkma sıklığının normal dağılımdan sapması"""
        total_draws = len(self.df)
        expected_freq = total_draws * 6 / 60  # Beklenen çıkma sayısı
        
        actual_freq = {num: 0 for num in range(1, 61)}
        for _, row in self.df.iterrows():
            for num in self.get_numbers(row):
                actual_freq[num] += 1
        
        # Standart sapma (binomial dağılım)
        p = 6/60  # bir çekilişte çıkma olasılığı
        std_dev = math.sqrt(total_draws * p * (1 - p))
        
        z_scores = {}
        for num in range(1, 61):
            if std_dev > 0:
                z_scores[num] = (actual_freq[num] - expected_freq) / std_dev
            else:
                z_scores[num] = 0
        
        return z_scores
    
    # ============================================================
    # 3. POISSON DAĞILIMI (Beklenti Analizi) - YENİ!
    # ============================================================
    
    def poisson_probability(self, k, lam):
        """Poisson olasılık hesaplama"""
        if lam <= 0:
            return 0
        return (math.exp(-lam) * (lam ** k)) / math.factorial(k)
    
    def calculate_poisson_scores(self):
        """Her sayının beklenen vs gerçekleşen çıkış farkı"""
        total_draws = len(self.df)
        lambda_val = total_draws * 6 / 60  # Beklenen çıkma sayısı
        
        actual_freq = {num: 0 for num in range(1, 61)}
        for _, row in self.df.iterrows():
            for num in self.get_numbers(row):
                actual_freq[num] += 1
        
        poisson_scores = {}
        for num in range(1, 61):
            # Bu sayının bu kadar az veya çok çıkma olasılığı
            prob_less = sum(self.poisson_probability(k, lambda_val) for k in range(actual_freq[num] + 1))
            
            # Vadesi dolmuşluk (negative binomial mantığı)
            if actual_freq[num] < lambda_val:
                # Beklenenden az çıkmış -> vadesi dolmuş olabilir
                poisson_scores[num] = (lambda_val - actual_freq[num]) / lambda_val
            else:
                poisson_scores[num] = 0.5  # Nötr
        
        return poisson_scores
    
    # ============================================================
    # 4. DELTA ANALİZİ (Sayılar arası farklar) - YENİ!
    # ============================================================
    
    def analyze_deltas(self):
        """Kazanan çekilişlerdeki sayılar arası farkların analizi"""
        all_deltas = []
        delta_freq = defaultdict(int)
        
        for _, row in self.df.iterrows():
            nums = sorted(self.get_numbers(row))
            for i in range(len(nums) - 1):
                delta = nums[i + 1] - nums[i]
                all_deltas.append(delta)
                delta_freq[delta] += 1
        
        # En sık görülen delta'lar
        common_deltas = [d for d, _ in sorted(delta_freq.items(), key=lambda x: x[1], reverse=True)[:5]]
        
        return {
            'common_deltas': common_deltas,
            'all_deltas': all_deltas,
            'avg_delta': sum(all_deltas) / len(all_deltas) if all_deltas else 5
        }
    
    def get_delta_compatible_numbers(self, k=20):
        """Delta analizine uygun sayılar"""
        delta_info = self.analyze_deltas()
        last_nums = sorted(self.get_numbers(self.df.iloc[-1]))
        
        compatible = set()
        for delta in delta_info['common_deltas'][:3]:
            for num in last_nums:
                candidate = num + delta
                if 1 <= candidate <= 60:
                    compatible.add(candidate)
                candidate = num - delta
                if 1 <= candidate <= 60:
                    compatible.add(candidate)
        
        return list(compatible)[:k]
    
    # ============================================================
    # 5. CO-OCCURRENCE MATRİSİ (Birlikte Çıkma) - YENİ!
    # ============================================================
    
    def _build_pair_cache(self):
        if self.pair_cache is not None:
            return self.pair_cache
        
        pair_count = defaultdict(int)
        for _, row in self.df.iterrows():
            nums = sorted(self.get_numbers(row))
            for pair in combinations(nums, 2):
                pair_count[pair] += 1
        
        self.pair_cache = pair_count
        return pair_cache
    
    def get_partners(self, num, top_n=5):
        """Bir sayının en çok birlikte çıktığı partnerleri"""
        pairs = self._build_pair_cache()
        partners = defaultdict(int)
        
        for (a, b), count in pairs.items():
            if a == num:
                partners[b] += count
            elif b == num:
                partners[a] += count
        
        sorted_partners = sorted(partners.items(), key=lambda x: x[1], reverse=True)
        return [p for p, _ in sorted_partners[:top_n]]
    
    # ============================================================
    # 6. TREND VE MOMENTUM ANALİZİ
    # ============================================================
    
    def get_trend_numbers(self, window=10, k=12):
        """Son window çekilişteki en formda sayılar"""
        recent_nums = []
        for idx in range(len(self.df) - window, len(self.df)):
            recent_nums.extend(self.get_numbers(self.df.iloc[idx]))
        counter = Counter(recent_nums)
        return [num for num, _ in counter.most_common(k)]
    
    def get_momentum_direction(self):
        """Sayısal ağırlık merkezinin kayma yönü"""
        recent_avg = []
        old_avg = []
        
        window = 10
        recent_window = self.df.iloc[-window:]
        older_window = self.df.iloc[-window*2:-window] if len(self.df) > window*2 else self.df.iloc[:window]
        
        for _, row in recent_window.iterrows():
            recent_avg.extend(self.get_numbers(row))
        for _, row in older_window.iterrows():
            old_avg.extend(self.get_numbers(row))
        
        recent_mean = sum(recent_avg) / len(recent_avg) if recent_avg else 30
        old_mean = sum(old_avg) / len(old_avg) if old_avg else 30
        
        return recent_mean - old_mean  # Pozitif = büyük sayılara kayıyor
    
    # ============================================================
    # 7. TEMEL PATTERN'LER (V2'den devral)
    # ============================================================
    
    def p_son7(self, k=12):
        recent_nums = []
        window = min(7, len(self.df))
        for idx in range(len(self.df) - window, len(self.df)):
            recent_nums.extend(self.get_numbers(self.df.iloc[idx]))
        counter = Counter(recent_nums)
        return [num for num, _ in counter.most_common(k)]
    
    def p_overdue(self, k=12):
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
    
    def p_low_entropy(self, k=12):
        def calculate_entropy(numbers):
            if not numbers:
                return 0
            ranges = [0] * 6
            for num in numbers:
                idx = (num - 1) // 10
                if 0 <= idx < 6:
                    ranges[idx] += 1
            entropy = 0
            total = len(numbers)
            for count in ranges:
                if count > 0:
                    p = count / total
                    entropy -= p * math.log2(p)
            return entropy
        
        all_nums = []
        for _, row in self.df.iterrows():
            nums = self.get_numbers(row)
            if calculate_entropy(nums) < 2.0:
                all_nums.extend(nums)
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(k)]
    
    def p_odd_draws(self, k=12):
        odd_indices = self.df.iloc[1::2]
        all_nums = []
        for _, row in odd_indices.iterrows():
            all_nums.extend(self.get_numbers(row))
        counter = Counter(all_nums)
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
    
    def p_top_pairs(self, k=12):
        pairs = self._build_pair_cache()
        top_pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)[:30]
        pair_numbers = []
        for pair, _ in top_pairs:
            pair_numbers.extend(pair)
        counter = Counter(pair_numbers)
        return [num for num, _ in counter.most_common(k)]


# ============================================================
# SÜPER HAVUZ ÜRETİCİ (28 Sayı - 4 Hücre)
# ============================================================

class SuperPoolGenerator:
    def __init__(self, df, get_numbers_func):
        self.df = df
        self.get_numbers = get_numbers_func
        self.pm = PatternMasterV3(df, get_numbers_func)
        
    def generate_super_pool(self):
        """4 hücreden oluşan 28 sayılık Süper Havuz"""
        
        print("\n" + "=" * 70)
        print("🔬 SÜPER HAVUZ ÜRETİLİYOR (V3 - İleri Analiz)")
        print("   Markov | Z-Score | Poisson | Delta | Co-occurrence")
        print("=" * 70)
        
        # ============================================================
        # HÜCRE A: TREND LİDERLERİ (Son 10 çekilişin en formda 7 sayısı)
        # ============================================================
        
        trend_nums = self.pm.get_trend_numbers(window=10, k=12)
        
        # Markov boost ile güçlendir
        markov_boost = self.pm.get_markov_boost()
        trend_scores = {}
        for num in trend_nums:
            trend_scores[num] = 1.0 + markov_boost.get(num, 0)
        
        sorted_trend = sorted(trend_scores.items(), key=lambda x: x[1], reverse=True)
        cell_a = [num for num, _ in sorted_trend[:7]]
        
        print(f"\n📊 HÜCRE A (Trend Liderleri) - 7 sayı:")
        print(f"   {cell_a}")
        print(f"   → Markov geçiş olasılığı yüksek sayılar")
        
        # ============================================================
        # HÜCRE B: MATEMATİKSEL BEKLENTİ (Overdue + Poisson + Z-Score)
        # ============================================================
        
        overdue_nums = self.pm.p_overdue(20)
        poisson_scores = self.pm.calculate_poisson_scores()
        z_scores = self.pm.calculate_z_scores()
        
        # Kombine skor
        expected_scores = {}
        for num in overdue_nums:
            score = 1.0
            score += poisson_scores.get(num, 0) * 2  # Poisson ağırlığı
            score += max(0, -z_scores.get(num, 0)) * 1.5  # Negatif Z-Score (az çıkmış)
            expected_scores[num] = score
        
        sorted_expected = sorted(expected_scores.items(), key=lambda x: x[1], reverse=True)
        cell_b = [num for num, _ in sorted_expected[:7]]
        
        print(f"\n📊 HÜCRE B (Matematiksel Beklenti) - 7 sayı:")
        print(f"   {cell_b}")
        print(f"   → Overdue + Poisson + Z-Score (vadesi dolmuş)")
        
        # ============================================================
        # HÜCRE C: KORELASYON/PARTNER (A ve B ile en çok eşleşen)
        # ============================================================
        
        combined_pool = set(cell_a + cell_b)
        partner_scores = defaultdict(int)
        
        for num in combined_pool:
            partners = self.pm.get_partners(num, top_n=3)
            for partner in partners:
                if partner not in combined_pool:
                    partner_scores[partner] += 1
        
        sorted_partners = sorted(partner_scores.items(), key=lambda x: x[1], reverse=True)
        cell_c = [num for num, _ in sorted_partners[:7]]
        
        # Eksik varsa delta uyumlu sayılarla tamamla
        if len(cell_c) < 7:
            delta_nums = self.pm.get_delta_compatible_numbers(10)
            for num in delta_nums:
                if num not in cell_c and num not in combined_pool:
                    cell_c.append(num)
                    if len(cell_c) >= 7:
                        break
        
        print(f"\n📊 HÜCRE C (Korelasyon/Partner) - 7 sayı:")
        print(f"   {cell_c}")
        print(f"   → A ve B ile en çok birlikte çıkan sayılar")
        
        # ============================================================
        # HÜCRE D: KAOS/SÜRPRİZ (Yüksek entropi + Düşük frekans)
        # ============================================================
        
        high_entropy_nums = self.pm.p_low_entropy(20)  # Düşük entropi aslında düzenli
        # Yüksek entropi için tam tersi
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_numbers(row))
        counter = Counter(all_nums)
        
        # En az çıkan 20 sayı
        least_frequent = [num for num, _ in sorted(counter.items(), key=lambda x: x[1])[:20]]
        
        # Momentum yönüne göre kaydır
        momentum = self.pm.get_momentum_direction()
        
        chaos_scores = {}
        for num in least_frequent:
            score = 1.0
            if momentum > 0 and num > 30:  # Büyük sayılara kayıyorsa
                score += 0.5
            elif momentum < 0 and num <= 30:
                score += 0.5
            chaos_scores[num] = score
        
        sorted_chaos = sorted(chaos_scores.items(), key=lambda x: x[1], reverse=True)
        cell_d = [num for num, _ in sorted_chaos[:7]]
        
        print(f"\n📊 HÜCRE D (Kaos/Sürpriz) - 7 sayı:")
        print(f"   {cell_d}")
        print(f"   → Düşük frekanslı, momentum uyumlu sürpriz sayılar")
        
        # ============================================================
        # SÜPER HAVUZ (28 Sayı)
        # ============================================================
        
        super_pool = cell_a + cell_b + cell_c + cell_d
        
        # Kırılma noktası kontrolü
        print("\n" + "-" * 70)
        print("🔬 KIRILMA NOKTASI ANALİZİ")
        print("-" * 70)
        
        # Her sayının kaç hücrede geçtiğini kontrol et (kesişim)
        all_cells = set(cell_a + cell_b + cell_c + cell_d)
        
        # Tekrar eden sayıları bul
        from collections import Counter as C
        pool_counter = C(cell_a + cell_b + cell_c + cell_d)
        duplicates = {num: count for num, count in pool_counter.items() if count > 1}
        
        if duplicates:
            print(f"   ⚠️ Kesişen sayılar: {duplicates}")
            print(f"   → Bunlar güçlü sinyal, havuzda kalmalı")
        
        # Benzersiz havuz
        unique_pool = list(set(super_pool))
        print(f"   📊 Benzersiz havuz: {len(unique_pool)} sayı")
        
        # Eksik varsa tamamla (delta uyumlu)
        if len(unique_pool) < 28:
            delta_nums = self.pm.get_delta_compatible_numbers(30)
            for num in delta_nums:
                if num not in unique_pool:
                    unique_pool.append(num)
                    if len(unique_pool) >= 28:
                        break
        
        super_pool = unique_pool[:28]
        
        # ============================================================
        # STATİSTİK RAPORU
        # ============================================================
        
        print("\n" + "-" * 70)
        print("🎯 SÜPER HAVUZ (28 SAYI) - FİNAL")
        print("-" * 70)
        
        # Sıralı olarak göster
        sorted_pool = sorted(super_pool)
        
        print(f"\n  {' '.join(f'{n:3d}' for n in sorted_pool[:14])}")
        print(f"  {' '.join(f'{n:3d}' for n in sorted_pool[14:])}")
        
        # Blok dağılımı
        blocks = {f"{i*10+1}-{(i+1)*10}": 0 for i in range(6)}
        for num in sorted_pool:
            block_idx = (num - 1) // 10
            block_key = f"{block_idx*10+1}-{(block_idx+1)*10}"
            blocks[block_key] = blocks.get(block_key, 0) + 1
        
        print(f"\n📊 Onluk Blok Dağılımı:")
        for block, count in blocks.items():
            bar = "█" * count
            print(f"  {block:12}: {count:2} sayı {bar}")
        
        # Tek/Çift dağılımı
        odds = [n for n in sorted_pool if n % 2 == 1]
        evens = [n for n in sorted_pool if n % 2 == 0]
        print(f"\n📊 Tek/Çift Dağılımı: {len(odds)} tek / {len(evens)} çift")
        
        # Büyük/Küçük dağılımı
        small = [n for n in sorted_pool if n <= 30]
        large = [n for n in sorted_pool if n > 30]
        print(f"📊 Büyük/Küçük Dağılımı: {len(small)} küçük (1-30) / {len(large)} büyük (31-60)")
        
        # Toplam istatistik
        print(f"\n📊 Havuz İstatistikleri:")
        print(f"  Min: {min(sorted_pool)} | Max: {max(sorted_pool)} | Range: {max(sorted_pool) - min(sorted_pool)}")
        print(f"  Ortalama: {sum(sorted_pool)/len(sorted_pool):.1f}")
        print(f"  Medyan: {sorted_pool[len(sorted_pool)//2]}")
        
        return {
            'super_pool': sorted_pool,
            'cell_a': cell_a,
            'cell_b': cell_b,
            'cell_c': cell_c,
            'cell_d': cell_d,
            'stats': {
                'odds_count': len(odds),
                'evens_count': len(evens),
                'small_count': len(small),
                'large_count': len(large),
                'avg': sum(sorted_pool)/len(sorted_pool),
                'range': max(sorted_pool) - min(sorted_pool)
            }
        }


# ============================================================
# PATTERN MASTER V3 ANA SINIF
# ============================================================

class SuperLotoPatternMasterV3:
    def __init__(self, excel_path="superloto.xlsx", sheet_name="s1"):
        self.excel_path = excel_path
        self.sheet_name = sheet_name
        self.df = None
        
    def load_data(self):
        loader = DataLoader(self.excel_path, self.sheet_name)
        self.df = loader.load()
        self.get_numbers = loader.get_numbers
        print(f"✅ Veri yüklendi: {len(self.df)} çekiliş")
        if len(self.df) > 0:
            print(f"📅 Aralık: {self.df['tarih'].min().strftime('%d.%m.%Y')} - {self.df['tarih'].max().strftime('%d.%m.%Y')}")
        return self.df
    
    def run(self):
        """Süper Havuz üret"""
        pool_gen = SuperPoolGenerator(self.df, self.get_numbers)
        result = pool_gen.generate_super_pool()
        return result
    
    def save_results(self, result):
        os.makedirs('outputs', exist_ok=True)
        
        # JSON kaydet
        with open('outputs/pattern_master_v3_super_pool.json', 'w', encoding='utf-8') as f:
            json.dump({
                'version': 'V3_SuperPool',
                'generated_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'last_draw': self.df['tarih'].max().strftime('%d.%m.%Y') if len(self.df) > 0 else None,
                'super_pool_28': result['super_pool'],
                'cell_a_trend': result['cell_a'],
                'cell_b_expectation': result['cell_b'],
                'cell_c_correlation': result['cell_c'],
                'cell_d_chaos': result['cell_d'],
                'statistics': result['stats']
            }, f, ensure_ascii=False, indent=2)
        
        # TXT rapor
        with open('outputs/pattern_master_v3_super_pool.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("🎯 SÜPER LOTO PATTERN MASTER V3 - SÜPER HAVUZ\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Oluşturulma: {pd.Timestamp.now().strftime('%d.%m.%Y %H:%M:%S')}\n")
            f.write(f"Son Çekiliş: {self.df['tarih'].max().strftime('%d.%m.%Y') if len(self.df) > 0 else '-'}\n")
            f.write(f"Toplam Analiz: {len(self.df)} çekiliş\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("🏆 SÜPER HAVUZ (28 SAYI)\n")
            f.write("-" * 70 + "\n\n")
            
            # 4x7 formatında göster
            f.write("  ")
            for i, num in enumerate(result['super_pool'][:14]):
                f.write(f"{num:3d} ")
                if i == 6:
                    f.write("\n  ")
            f.write("\n  ")
            for i, num in enumerate(result['super_pool'][14:]):
                f.write(f"{num:3d} ")
                if i == 6:
                    f.write("\n  ")
            f.write("\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("🔬 4 HÜCRE YAPISI\n")
            f.write("-" * 70 + "\n\n")
            
            f.write("HÜCRE A (Trend Liderleri):\n")
            f.write(f"  {result['cell_a']}\n")
            f.write("  → Son 10 çekilişin en formda sayıları + Markov\n\n")
            
            f.write("HÜCRE B (Matematiksel Beklenti):\n")
            f.write(f"  {result['cell_b']}\n")
            f.write("  → Overdue + Poisson + Z-Score\n\n")
            
            f.write("HÜCRE C (Korelasyon/Partner):\n")
            f.write(f"  {result['cell_c']}\n")
            f.write("  → A ve B ile en çok birlikte çıkanlar\n\n")
            
            f.write("HÜCRE D (Kaos/Sürpriz):\n")
            f.write(f"  {result['cell_d']}\n")
            f.write("  → Düşük frekanslı, momentum uyumlu\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("📊 İSTATİSTİKLER\n")
            f.write("-" * 70 + "\n\n")
            f.write(f"  Tek/Çift: {result['stats']['odds_count']}/{result['stats']['evens_count']}\n")
            f.write(f"  Küçük/Büyük: {result['stats']['small_count']}/{result['stats']['large_count']}\n")
            f.write(f"  Ortalama: {result['stats']['avg']:.1f}\n")
            f.write(f"  Aralık: {result['stats']['range']}\n")
        
        print(f"\n💾 Kaydedildi: outputs/pattern_master_v3_super_pool.json")
        print(f"💾 Kaydedildi: outputs/pattern_master_v3_super_pool.txt")
    
    def print_report(self, result):
        print("\n" + "=" * 70)
        print("🎯 SÜPER LOTO PATTERN MASTER V3 - SÜPER HAVUZ RAPORU")
        print("   Markov | Z-Score | Poisson | Delta | Co-occurrence")
        print("=" * 70)
        
        print("\n🏆 SÜPER HAVUZ (28 SAYI):")
        print("-" * 70)
        
        sorted_pool = result['super_pool']
        print(f"\n  {' '.join(f'{n:3d}' for n in sorted_pool[:14])}")
        print(f"  {' '.join(f'{n:3d}' for n in sorted_pool[14:])}")
        
        print("\n" + "-" * 70)
        print("🔬 4 HÜCRE YAPISI (Nokta Atışı)")
        print("-" * 70)
        
        print(f"\n  HÜCRE A (Trend):      {result['cell_a']}")
        print(f"  HÜCRE B (Beklenti):   {result['cell_b']}")
        print(f"  HÜCRE C (Partner):    {result['cell_c']}")
        print(f"  HÜCRE D (Sürpriz):    {result['cell_d']}")
        
        print("\n" + "-" * 70)
        print("📊 İSTATİSTİKLER")
        print("-" * 70)
        print(f"  Tek/Çift: {result['stats']['odds_count']}/{result['stats']['evens_count']}")
        print(f"  Küçük/Büyük: {result['stats']['small_count']}/{result['stats']['large_count']}")
        print(f"  Ortalama: {result['stats']['avg']:.1f}")
        print(f"  Aralık: {result['stats']['range']}")
        
        print("\n" + "-" * 70)
        print("⚠️ NOT: Bu 28 sayılık SÜPER HAVUZ, 4 farklı matematiksel")
        print("   yöntemin birleşimidir. Kombinasyona gerek yok.")
        print("   Hedef: 28 sayıda 6/6 isabet!")
        print("=" * 70)
        
        return result


# ============================================================
# ANA ÇALIŞTIRMA
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("🚀 SÜPER LOTO PATTERN MASTER V3")
    print("   Markov Zinciri | Z-Score | Poisson | Delta | Co-occurrence")
    print("   TEK BOT - NOKTATAŞI - 28 SAYILIK SÜPER HAVUZ")
    print("=" * 70)
    
    bot = SuperLotoPatternMasterV3()
    bot.load_data()
    result = bot.run()
    bot.print_report(result)
    bot.save_results(result)
    
    print("\n✅ TAMAMLANDI!")

if __name__ == "__main__":
    main()
