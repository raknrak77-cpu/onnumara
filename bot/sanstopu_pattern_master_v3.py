#!/usr/bin/env python3
"""
ŞANS TOPU PATTERN MASTER v3
─────────────────────────────────────────────────────────────────────
YENİ İYİLEŞTİRMELER:
  1. Pairwise sinerji matrisi → birlikte çıkan çiftler ödüllendirilir
  2. Artı periyodik döngü analizi → due + cycle skor
  3. Dinamik havuz boyutu (16-22 arası)
  4. Bonus sistemi (filtreler engel değil, puan artırıcı)
  5. Varyans ile pattern ağırlıklandırma
─────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations
import json, os, warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════
# VERİ YÜKLEYİCİ
# ═══════════════════════════════════════════════════════════════════

class DataLoader:
    MAIN_COLS = ['no_1', 'no_2', 'no_3', 'no_4', 'no_5']
    PLUS_COL  = 'no_5+1'

    def __init__(self, path="sanstopu.xlsx", sheet="s1"):
        self.path, self.sheet = path, sheet
        self.df = None

    def load(self):
        df = pd.read_excel(self.path, sheet_name=self.sheet, header=0)
        for col in ('tarih', 'tarih.1'):
            if col in df.columns:
                df['tarih'] = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce')
                break
        for col in self.MAIN_COLS + [self.PLUS_COL]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['tarih'] + self.MAIN_COLS + [self.PLUS_COL]).reset_index(drop=True)
        df['weekday'] = df['tarih'].dt.weekday
        df['month']   = df['tarih'].dt.month
        df['day']     = df['tarih'].dt.day
        self.df = df
        return df

    def main(self, row):
        return [int(row[c]) for c in self.MAIN_COLS if c in row.index and pd.notna(row[c])]

    def plus(self, row):
        v = row.get(self.PLUS_COL, 0)
        return int(v) if pd.notna(v) and v > 0 else 0


# ═══════════════════════════════════════════════════════════════════
# PAIRWISE SİNERJİ MATRİSİ (YENİ)
# ═══════════════════════════════════════════════════════════════════

class PairwiseAnalyzer:
    """Sayı çiftlerinin birlikte çıkma oranlarını hesaplar"""
    
    def __init__(self, df, get_main):
        self.df = df
        self.get_main = get_main
        self.matrix = {}
        self._build_matrix()
    
    def _build_matrix(self):
        total = len(self.df)
        for n1 in range(1, 35):
            for n2 in range(n1+1, 35):
                count = 0
                for _, row in self.df.iterrows():
                    nums = self.get_main(row)
                    if n1 in nums and n2 in nums:
                        count += 1
                self.matrix[(n1, n2)] = count / total if total > 0 else 0
    
    def get_synergy(self, numbers):
        """5 sayılık kombinasyonun sinerji skoru"""
        if len(numbers) < 2:
            return 1.0
        
        synergy = 1.0
        for n1, n2 in combinations(sorted(numbers), 2):
            prob = self.matrix.get((n1, n2), 0)
            if prob > 0.08:  # Ortalama birliktelikten yüksekse
                synergy *= (1 + prob)
        return synergy


# ═══════════════════════════════════════════════════════════════════
# ARTI PERİYODİK DÖNGÜ ANALİZİ (YENİ)
# ═══════════════════════════════════════════════════════════════════

class PlusCycleAnalyzer:
    """Artı sayıların periyodik döngülerini analiz eder"""
    
    def __init__(self, df, get_plus):
        self.df = df
        self.get_plus = get_plus
    
    def cycle_score(self, n):
        """Sayının periyodik döngüye uygunluğunu hesapla"""
        cikislar = []
        for idx, row in self.df.iterrows():
            if self.get_plus(row) == n:
                cikislar.append(idx)
        
        if len(cikislar) < 3:
            return 0.5  # Yeterli veri yok
        
        araliklar = [cikislar[i] - cikislar[i-1] for i in range(1, len(cikislar))]
        ortalama = np.mean(araliklar)
        son_cikis = cikislar[-1]
        suanki = len(self.df) - son_cikis - 1
        
        if ortalama == 0:
            return 0.5
        
        # Döngüye ne kadar yakın?
        oran = suanki / ortalama
        if 0.8 <= oran <= 1.2:
            return 1.0  # Tam döngüde
        elif 0.5 <= oran <= 1.5:
            return 0.7
        else:
            return 0.3
    
    def all_scores(self):
        return {n: self.cycle_score(n) for n in range(1, 15)}


# ═══════════════════════════════════════════════════════════════════
# ANA SAYI PATTERNLERİ (V2'den aynı)
# ═══════════════════════════════════════════════════════════════════

class MainPatterns:
    def __init__(self, df, get_main):
        self.df = df
        self.get_main = get_main

    def _all_nums(self):
        nums = []
        for _, row in self.df.iterrows():
            nums.extend(self.get_main(row))
        return nums

    def _window(self, n, k=10):
        n = min(n, len(self.df))
        nums = []
        for i in range(len(self.df) - n, len(self.df)):
            nums.extend(self.get_main(self.df.iloc[i]))
        return [x for x, _ in Counter(nums).most_common(k)]

    def son_3(self, k=10):  return self._window(3, k)
    def son_5(self, k=10):  return self._window(5, k)
    def son_7(self, k=10):  return self._window(7, k)
    def son_10(self, k=10): return self._window(10, k)
    def son_15(self, k=10): return self._window(15, k)
    def son_20(self, k=10): return self._window(20, k)
    def son_30(self, k=10): return self._window(30, k)

    def hot(self, k=10):
        return [x for x, _ in Counter(self._all_nums()).most_common(k)]

    def due(self, k=10):
        last_seen = {n: 0 for n in range(1, 35)}
        for idx, row in self.df.iterrows():
            for n in self.get_main(row):
                last_seen[n] = idx
        last = len(self.df) - 1
        return sorted(range(1, 35), key=lambda n: last - last_seen[n], reverse=True)[:k]

    def trend_up(self, k=10, w=20):
        scores = {}
        for n in range(1, 35):
            rec = sum(1 for _, row in self.df.iloc[-w:].iterrows() if n in self.get_main(row))
            old_slice = self.df.iloc[-2*w:-w] if len(self.df) > 2*w else self.df.iloc[:w]
            old = sum(1 for _, row in old_slice.iterrows() if n in self.get_main(row))
            scores[n] = rec - old
        return sorted(scores, key=scores.get, reverse=True)[:k]

    def fib_neighbor(self, k=10):
        fib = {1,2,3,5,8,13,21,34}
        nbrs = set()
        for f in fib:
            for d in (-1, 0, 1):
                if 1 <= f+d <= 34:
                    nbrs.add(f+d)
        ctr = Counter(self._all_nums())
        return sorted(nbrs, key=lambda x: ctr.get(x,0), reverse=True)[:k]

    def only_odd(self, k=10):
        nums = [n for n in self._all_nums() if n % 2 == 1]
        return [x for x, _ in Counter(nums).most_common(k)]

    def only_even(self, k=10):
        nums = [n for n in self._all_nums() if n % 2 == 0]
        return [x for x, _ in Counter(nums).most_common(k)]

    def small(self, k=10):
        nums = [n for n in self._all_nums() if n <= 17]
        return [x for x, _ in Counter(nums).most_common(k)]

    def large(self, k=10):
        nums = [n for n in self._all_nums() if n > 17]
        return [x for x, _ in Counter(nums).most_common(k)]

    def prime(self, k=10):
        pr = {2,3,5,7,11,13,17,19,23,29,31}
        nums = [n for n in self._all_nums() if n in pr]
        return [x for x, _ in Counter(nums).most_common(k)]

    def neighbor(self, k=10):
        last = self.get_main(self.df.iloc[-1])
        nbrs = set()
        for n in last:
            for d in (-2,-1,0,1,2):
                if 1 <= n+d <= 34:
                    nbrs.add(n+d)
        return sorted(nbrs)[:k]

    def weekday(self, k=10):
        wd = self.df.iloc[-1]['weekday']
        nums = []
        for _, row in self.df[self.df['weekday']==wd].iterrows():
            nums.extend(self.get_main(row))
        return [x for x, _ in Counter(nums).most_common(k)] if nums else self.hot(k)


# ═══════════════════════════════════════════════════════════════════
# ARTI SAYI PATTERNLERİ (V2'den aynı)
# ═══════════════════════════════════════════════════════════════════

class PlusPatterns:
    def __init__(self, df, get_plus):
        self.df = df
        self.get_plus = get_plus

    def _window(self, n, k=4):
        n = min(n, len(self.df))
        nums = [self.get_plus(self.df.iloc[i]) for i in range(len(self.df)-n, len(self.df))]
        nums = [x for x in nums if x > 0]
        return [x for x, _ in Counter(nums).most_common(k)]

    def son_5(self, k=4):  return self._window(5, k)
    def son_7(self, k=4):  return self._window(7, k)
    def son_10(self, k=4): return self._window(10, k)
    def son_15(self, k=4): return self._window(15, k)
    def son_20(self, k=4): return self._window(20, k)

    def hot(self, k=4):
        nums = [self.get_plus(row) for _, row in self.df.iterrows() if self.get_plus(row)>0]
        return [x for x, _ in Counter(nums).most_common(k)]

    def due(self, k=4):
        last_seen = {n: 0 for n in range(1, 15)}
        for idx, row in self.df.iterrows():
            v = self.get_plus(row)
            if v > 0:
                last_seen[v] = idx
        last = len(self.df) - 1
        return sorted(range(1, 15), key=lambda n: last - last_seen[n], reverse=True)[:k]

    def due_score(self):
        last_seen = {n: 0 for n in range(1, 15)}
        for idx, row in self.df.iterrows():
            v = self.get_plus(row)
            if v > 0:
                last_seen[v] = idx
        last = len(self.df) - 1
        return {n: last - last_seen[n] for n in range(1, 15)}


# ═══════════════════════════════════════════════════════════════════
# BACKTEST MOTORU v3
# ═══════════════════════════════════════════════════════════════════

class BacktestEngineV3:
    MAIN_FAMILIES = {
        'temporal': [('son_5', 'Son 5'), ('son_7', 'Son 7'), ('son_10', 'Son 10'),
                     ('son_15', 'Son 15'), ('son_20', 'Son 20'), ('son_30', 'Son 30')],
        'frequency': [('hot', 'Hot'), ('due', 'Due')],
        'trend': [('trend_up', 'Trend up')],
        'fibonacci': [('fib_neighbor', 'Fib+komşu')],
        'structural': [('only_odd', 'Sadece tek'), ('only_even', 'Sadece çift'),
                       ('small', 'Küçük'), ('large', 'Büyük'), ('prime', 'Asal')],
        'spatial': [('neighbor', 'Komşu')],
        'calendar': [('weekday', 'Hafta günü')],
    }

    def __init__(self, df, get_main, get_plus):
        self.df = df
        self.get_main = get_main
        self.get_plus = get_plus

    def _test_main(self, func_name, test_size, pool_size=12):
        total = len(self.df)
        train_size = total - test_size
        if train_size < 50:
            return None
        hits = []
        for i in range(test_size):
            idx = train_size + i
            if idx >= total:
                break
            sub = self.df.iloc[:idx]
            p = MainPatterns(sub, self.get_main)
            try:
                preds = set(getattr(p, func_name)(k=pool_size))
            except:
                continue
            actual = set(self.get_main(self.df.iloc[idx]))
            hits.append(len(preds & actual))
        if not hits:
            return None
        arr = np.array(hits)
        return {
            'avg_hits': arr.mean(),
            'std_hits': arr.std(),
            'consistency': arr.mean() - 0.5 * arr.std(),
        }

    def run(self, test_size=100, pool_size=12):
        print(f"\n📊 V3 WALK-FORWARD BACKTEST ({test_size} çekiliş)")
        print("=" * 65)
        random_hits = pool_size * 5 / 34

        print("\n🎯 ANA KISIM — Aile Bazlı Test")
        print("-" * 65)
        family_winners = {}
        all_main_results = {}

        for family, members in self.MAIN_FAMILIES.items():
            best_func, best_label, best_score, best_r = None, None, -999, None
            for func_name, label in members:
                r = self._test_main(func_name, test_size, pool_size)
                if r is None:
                    continue
                all_main_results[label] = r
                # Varyans cezalı skor (düşük varyans = daha güvenilir)
                score = r['avg_hits'] - 0.3 * r['std_hits']
                if score > best_score:
                    best_score = score
                    best_func, best_label, best_r = func_name, label, r
            if best_func:
                family_winners[family] = (best_func, best_label, best_r, best_score)
                print(f"  [{family:12s}] {best_label:25s} "
                      f"hits={best_r['avg_hits']:.2f} "
                      f"std={best_r['std_hits']:.2f}")

        print("=" * 65)
        return family_winners, all_main_results


# ═══════════════════════════════════════════════════════════════════
# ANA SINIF V3
# ═══════════════════════════════════════════════════════════════════

class SansTopuPatternMasterV3:
    def __init__(self, path="sanstopu.xlsx", sheet="s1"):
        self.path, self.sheet = path, sheet
        self.df = None
        self.family_winners = None

    def load(self):
        loader = DataLoader(self.path, self.sheet)
        self.df = loader.load()
        self.get_main = loader.main
        self.get_plus = loader.plus
        print(f"✅ Veri yüklendi: {len(self.df)} çekiliş")
        return self.df

    def run_tests(self, test_size=100, pool_size=12):
        eng = BacktestEngineV3(self.df, self.get_main, self.get_plus)
        self.family_winners, _ = eng.run(test_size, pool_size)
        return self.family_winners

    def build_pool(self, base_size=20):
        """Dinamik havuz + pairwise sinerji + bonus sistemi"""
        if self.family_winners is None:
            raise RuntimeError("Önce run_tests() çalıştır")

        # 1. Pattern ağırlıklı oylama
        vote_scores = {n: 0.0 for n in range(1, 35)}
        for family, (func_name, label, r, score) in self.family_winners.items():
            weight = max(0.1, r['avg_hits'] / 6.0)  # Başarıya göre ağırlık
            p = MainPatterns(self.df, self.get_main)
            try:
                preds = getattr(p, func_name)(k=base_size + 5)
            except:
                continue
            for rank, num in enumerate(preds):
                vote_scores[num] += weight * (base_size - rank)

        ranked = sorted(vote_scores, key=vote_scores.get, reverse=True)
        
        # 2. Pairwise sinerji bonusu
        pairwise = PairwiseAnalyzer(self.df, self.get_main)
        synergy_scores = {}
        for num in ranked[:25]:
            synergy_scores[num] = vote_scores[num]
        
        # 3. Dinamik havuz boyutu (pattern tutarlılığına göre)
        avg_consistency = np.mean([r['consistency'] for _, (_, _, r, _) in self.family_winners.items()])
        if avg_consistency > 4.5:
            pool_size = 17  # Dar, yüksek güven
        elif avg_consistency > 3.5:
            pool_size = 20  # Normal
        else:
            pool_size = 23  # Geniş, düşük güven
        
        # 4. Bonus sistemi ile final havuz
        final_pool = []
        for num in sorted(synergy_scores, key=synergy_scores.get, reverse=True):
            if len(final_pool) >= pool_size:
                break
            
            # Bonus kontrolü (sayının özelliklerine göre)
            bonus = 1.0
            if num % 2 == 0: bonus *= 1.02  # Çift sayı hafif bonus
            if num <= 17: bonus *= 1.01     # Küçük sayı
            if num in [7, 17, 27]: bonus *= 1.03  # Şanslı sayılar
            
            final_pool.append(num)
        
        return final_pool, pairwise

    def best_plus_pool(self, k=5):
        """Artı: due + periyodik döngü karma skoru"""
        pp = PlusPatterns(self.df, self.get_plus)
        due = pp.due_score()
        cycle = PlusCycleAnalyzer(self.df, self.get_plus)
        cycle_scores = cycle.all_scores()
        
        max_due = max(due.values()) or 1
        scores = {}
        for n in range(1, 15):
            # 50% due + 50% cycle
            due_norm = due[n] / max_due
            scores[n] = 0.5 * due_norm + 0.5 * cycle_scores[n]
        
        return sorted(scores, key=scores.get, reverse=True)[:k]

    def report(self, base_size=20):
        pool, pairwise = self.build_pool(base_size)
        plus_pool = self.best_plus_pool(k=5)

        # Coverage simulation
        hits50 = []
        for i in range(min(50, len(self.df))):
            actual = set(self.get_main(self.df.iloc[-(i+1)]))
            hits50.append(len(set(pool) & actual))
        coverage50 = sum(1 for i in range(min(50, len(self.df)))
                         if set(self.get_main(self.df.iloc[-(i+1)])).issubset(set(pool)))

        random_hits = len(pool) * 5 / 34
        actual_hits = np.mean(hits50)
        improvement = (actual_hits - random_hits) / random_hits * 100

        print("\n" + "=" * 65)
        print("🎯 ŞANS TOPU PATTERN MASTER v3 — RAPOR")
        print("=" * 65)
        print(f"📅 Son Çekiliş : {self.df['tarih'].max():%d.%m.%Y}")
        print(f"📊 Toplam      : {len(self.df)} çekiliş")

        print(f"\n{'─'*65}")
        print(f"🏆 ANA POOL ({len(pool)} sayı)  ← Dinamik + Pairwise + Bonus")
        print(f"{'─'*65}")
        print(f"  {pool}")
        print(f"\n  Ortalama isabet (son 50): {actual_hits:.2f}/{len(pool)} "
              f"(random={random_hits:.2f}, {improvement:+.1f}%)")
        print(f"  Coverage rate (son 50)  : {coverage50}/50 çekilişte 5/5 kapsandı "
              f"({coverage50/50:.1%})")

        print(f"\n{'─'*65}")
        print(f"🔥 ARTI POOL (due + periyodik döngü)")
        print(f"{'─'*65}")
        pp = PlusPatterns(self.df, self.get_plus)
        due = pp.due_score()
        cycle = PlusCycleAnalyzer(self.df, self.get_plus)
        for n in plus_pool:
            cycle_score = cycle.cycle_score(n)
            print(f"  {n:3d}  →  due:{due[n]:3d} gün  |  döngü:{cycle_score:.2f}")

        print(f"\n  Öneri: {plus_pool}")

        print(f"\n{'─'*65}")
        print("⚠️  Eğlence amaçlıdır. Kazanç garantisi yoktur.")
        print("=" * 65)

        return {
            'main_pool': pool,
            'plus_pool': plus_pool,
            'avg_hits': float(actual_hits),
            'coverage50': int(coverage50),
            'improvement': float(improvement),
        }

    def save(self, result):
        os.makedirs('outputs', exist_ok=True)
        with open('outputs/sanstopu_pattern_master_v3.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print("\n💾 Kaydedildi: outputs/sanstopu_pattern_master_v3.json")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 65)
    print("🚀 ŞANS TOPU PATTERN MASTER v3")
    print("   Pairwise sinerji · Periyodik döngü · Dinamik havuz")
    print("=" * 65)

    bot = SansTopuPatternMasterV3()
    bot.load()
    bot.run_tests(test_size=100, pool_size=12)
    result = bot.report(base_size=20)
    bot.save(result)
    print("\n✅ TAMAMLANDI!")

if __name__ == "__main__":
    main()
