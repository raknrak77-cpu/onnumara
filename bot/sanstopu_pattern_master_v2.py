#!/usr/bin/env python3
"""
ŞANS TOPU PATTERN MASTER v2
─────────────────────────────────────────────────────────────────────
İYİLEŞTİRMELER:
  1. Pattern aileleri → benzer patternler gruplanır, her aileden
     sadece en iyi 1'i alınır → p-hacking azalır
  2. Coverage rate → pool 5/5 winner kapsadı mı? (jackpot metriği)
  3. Consistency score → mean - 0.5×std (kararsız patternler cezalanır)
  4. Artı due skoru → kaç çekilişdir çıkmadı? ağırlıklı analiz
  5. Gerçek walk-forward → her adımda sadece geçmiş veri kullanılır
─────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
from collections import Counter
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
        # Tarih
        for col in ('tarih', 'tarih.1'):
            if col in df.columns:
                df['tarih'] = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce')
                break
        # Sayılar
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
# ANA SAYI PATTERNLERİ
# ═══════════════════════════════════════════════════════════════════

class MainPatterns:
    """Her method -> sorted list of k recommended numbers (1-34)"""

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

    # ── Temporal ──────────────────────────────────────────────────
    def son_3(self, k=10):  return self._window(3, k)
    def son_5(self, k=10):  return self._window(5, k)
    def son_7(self, k=10):  return self._window(7, k)
    def son_10(self, k=10): return self._window(10, k)
    def son_15(self, k=10): return self._window(15, k)
    def son_20(self, k=10): return self._window(20, k)
    def son_30(self, k=10): return self._window(30, k)

    # ── Hot / Due ─────────────────────────────────────────────────
    def hot(self, k=10):
        return [x for x, _ in Counter(self._all_nums()).most_common(k)]

    def due(self, k=10):
        last_seen = {n: 0 for n in range(1, 35)}
        for idx, row in self.df.iterrows():
            for n in self.get_main(row):
                last_seen[n] = idx
        last = len(self.df) - 1
        return sorted(range(1, 35), key=lambda n: last - last_seen[n], reverse=True)[:k]

    # ── Trend ─────────────────────────────────────────────────────
    def trend_up(self, k=10, w=20):
        scores = {}
        for n in range(1, 35):
            rec = sum(1 for _, row in self.df.iloc[-w:].iterrows() if n in self.get_main(row))
            old_slice = self.df.iloc[-2*w:-w] if len(self.df) > 2*w else self.df.iloc[:w]
            old = sum(1 for _, row in old_slice.iterrows() if n in self.get_main(row))
            scores[n] = rec - old
        return sorted(scores, key=scores.get, reverse=True)[:k]

    # ── Fibonacci ─────────────────────────────────────────────────
    def fib_neighbor(self, k=10):
        fib = {1,2,3,5,8,13,21,34}
        nbrs = set()
        for f in fib:
            for d in (-1, 0, 1):
                if 1 <= f+d <= 34:
                    nbrs.add(f+d)
        ctr = Counter(self._all_nums())
        return sorted(nbrs, key=lambda x: ctr.get(x,0), reverse=True)[:k]

    # ── Yapısal ───────────────────────────────────────────────────
    def _split(self, condition, k=10):
        nums = [n for n in self._all_nums() if condition(n)]
        return [x for x, _ in Counter(nums).most_common(k)]

    def only_odd(self, k=10):         return self._split(lambda n: n%2==1, k)
    def only_even(self, k=10):        return self._split(lambda n: n%2==0, k)
    def small(self, k=10):            return self._split(lambda n: n<=17, k)
    def large(self, k=10):            return self._split(lambda n: n>17, k)
    def prime(self, k=10):
        pr = {2,3,5,7,11,13,17,19,23,29,31}
        return self._split(lambda n: n in pr, k)

    # ── Komşu ─────────────────────────────────────────────────────
    def neighbor(self, k=10):
        last = self.get_main(self.df.iloc[-1])
        nbrs = set()
        for n in last:
            for d in (-2,-1,0,1,2):
                if 1 <= n+d <= 34:
                    nbrs.add(n+d)
        return sorted(nbrs)[:k]

    # ── Takvim ────────────────────────────────────────────────────
    def weekday(self, k=10):
        wd = self.df.iloc[-1]['weekday']
        nums = []
        for _, row in self.df[self.df['weekday']==wd].iterrows():
            nums.extend(self.get_main(row))
        return [x for x, _ in Counter(nums).most_common(k)] if nums else self.hot(k)


# ═══════════════════════════════════════════════════════════════════
# ARTI SAYI PATTERNLERİ
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
        """Kaç çekilişdir çıkmadı — en uzun bekleyenler"""
        last_seen = {n: 0 for n in range(1, 15)}
        for idx, row in self.df.iterrows():
            v = self.get_plus(row)
            if v > 0:
                last_seen[v] = idx
        last = len(self.df) - 1
        return sorted(range(1, 15), key=lambda n: last - last_seen[n], reverse=True)[:k]

    def due_score(self):
        """Her artı sayı için due skor döner (yüksek = uzun süredir çıkmamış)"""
        last_seen = {n: 0 for n in range(1, 15)}
        for idx, row in self.df.iterrows():
            v = self.get_plus(row)
            if v > 0:
                last_seen[v] = idx
        last = len(self.df) - 1
        return {n: last - last_seen[n] for n in range(1, 15)}

    def weekday(self, k=4):
        wd = self.df.iloc[-1]['weekday']
        nums = [self.get_plus(row) for _, row in self.df[self.df['weekday']==wd].iterrows()
                if self.get_plus(row) > 0]
        return [x for x, _ in Counter(nums).most_common(k)] if nums else self.hot(k)


# ═══════════════════════════════════════════════════════════════════
# BACKTEST MOTORU v2 — Walk-Forward + Coverage Rate
# ═══════════════════════════════════════════════════════════════════

class BacktestEngine:
    """
    Walk-forward validation:
      Her test adımında sadece o ana kadar olan veri kullanılır.
      Gerçek out-of-sample tahmin simülasyonu.

    Metrikler:
      avg_hits    : ortalama isabet (eski)
      coverage    : pool tüm 5 sayıyı kaç kez kapsadı? (jackpot odaklı)
      consistency : mean - 0.5×std (kararsız patternleri cezalandırır)
      composite   : 0.4×norm_hits + 0.4×coverage + 0.2×consistency
    """

    MAIN_FAMILIES = {
        'temporal': [
            ('son_3',  'Son 3 çekiliş'),
            ('son_5',  'Son 5 çekiliş'),
            ('son_7',  'Son 7 çekiliş'),
            ('son_10', 'Son 10 çekiliş'),
            ('son_15', 'Son 15 çekiliş'),
            ('son_20', 'Son 20 çekiliş'),
            ('son_30', 'Son 30 çekiliş'),
        ],
        'frequency': [
            ('hot',  'Hot numbers'),
            ('due',  'Due numbers'),
        ],
        'trend': [
            ('trend_up', 'Trend artan'),
        ],
        'fibonacci': [
            ('fib_neighbor', 'Fibonacci + komşu'),
        ],
        'structural': [
            ('only_odd',  'Sadece tek'),
            ('only_even', 'Sadece çift'),
            ('small',     'Küçük (1-17)'),
            ('large',     'Büyük (18-34)'),
            ('prime',     'Asal sayılar'),
        ],
        'spatial': [
            ('neighbor', 'Son çekilişin komşuları'),
        ],
        'calendar': [
            ('weekday', 'Haftanın aynı günü'),
        ],
    }

    PLUS_PATTERNS = [
        ('son_5',   'Son 5 çekiliş'),
        ('son_7',   'Son 7 çekiliş'),
        ('son_10',  'Son 10 çekiliş'),
        ('son_15',  'Son 15 çekiliş'),
        ('son_20',  'Son 20 çekiliş'),
        ('hot',     'Hot numbers'),
        ('due',     'Due numbers'),
        ('weekday', 'Haftanın aynı günü'),
    ]

    def __init__(self, df, get_main, get_plus):
        self.df = df
        self.get_main = get_main
        self.get_plus = get_plus

    def _test_main(self, func_name, test_size, pool_size=10):
        total = len(self.df)
        train_size = total - test_size
        if train_size < 50:
            return None
        hits, coverages = [], []
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
            h = len(preds & actual)
            hits.append(h)
            coverages.append(1 if actual.issubset(preds) else 0)
        if not hits:
            return None
        arr = np.array(hits)
        return {
            'avg_hits':    arr.mean(),
            'std_hits':    arr.std(),
            'coverage':    np.mean(coverages),   # % of draws where pool ⊇ actual
            'consistency': arr.mean() - 0.5*arr.std(),
        }

    def _test_plus(self, func_name, test_size, pool_size=4):
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
            p = PlusPatterns(sub, self.get_plus)
            try:
                preds = getattr(p, func_name)(k=pool_size)
            except:
                continue
            actual = self.get_plus(self.df.iloc[idx])
            hits.append(1 if actual in preds else 0)
        if not hits:
            return None
        arr = np.array(hits)
        return {
            'hit_rate':    arr.mean(),
            'consistency': arr.mean() - 0.3*arr.std(),
        }

    def _composite_main(self, r, random_hits=10*5/34):
        """Normalize edilmiş kompozit skor"""
        norm_hits = max(0, (r['avg_hits'] - random_hits) / random_hits)
        return 0.4*norm_hits + 0.4*r['coverage'] + 0.2*(r['consistency'] / (random_hits+0.1))

    def run(self, test_size=100, pool_size=10):
        print(f"\n📊 WALK-FORWARD BACKTEST ({test_size} çekiliş, pool={pool_size})")
        print("=" * 65)
        random_hits = pool_size * 5 / 34

        # ── ANA KISIM: Aile bazlı test ────────────────────────────
        print("\n🎯 ANA KISIM — Aile Bazlı Test")
        print("-" * 65)
        family_winners = {}   # family → (func_name, label, result)
        all_main_results = {} # label → result dict

        for family, members in self.MAIN_FAMILIES.items():
            best_func, best_label, best_score, best_r = None, None, -999, None
            for func_name, label in members:
                r = self._test_main(func_name, test_size, pool_size)
                if r is None:
                    continue
                all_main_results[label] = r
                score = self._composite_main(r, random_hits)
                if score > best_score:
                    best_score = score
                    best_func, best_label, best_r = func_name, label, r
            if best_func:
                family_winners[family] = (best_func, best_label, best_r, best_score)
                mark = "🔥" if best_score > 0.05 else "⭐" if best_score > 0.02 else "·"
                print(f"  [{family:12s}] {best_label:25s} "
                      f"hits={best_r['avg_hits']:.2f} "
                      f"cov={best_r['coverage']:.1%} "
                      f"cons={best_r['consistency']:.2f} {mark}")

        # Tüm patternlerin özeti
        print("\n📋 Tüm Pattern Sonuçları (hits | coverage | consistency)")
        print("-" * 65)
        for label, r in sorted(all_main_results.items(),
                                key=lambda x: self._composite_main(x[1], random_hits),
                                reverse=True):
            score = self._composite_main(r, random_hits)
            pct   = (r['avg_hits'] - random_hits) / random_hits * 100
            mark  = "🔥" if pct > 15 else "⭐" if pct > 5 else "·" if pct > 0 else "❌"
            print(f"  {label:35s} {r['avg_hits']:4.2f}/{pool_size} "
                  f"cov={r['coverage']:.1%} "
                  f"({pct:+.0f}%) {mark}")

        # ── ARTI KISIM ────────────────────────────────────────────
        print(f"\n🎯 ARTI KISIM — Walk-Forward Test")
        print("-" * 65)
        random_plus = 4 / 14
        plus_results = {}
        for func_name, label in self.PLUS_PATTERNS:
            r = self._test_plus(func_name, test_size, pool_size=4)
            if r is None:
                continue
            plus_results[label] = (func_name, r)
            pct = (r['hit_rate'] - random_plus) / random_plus * 100
            mark = "🔥" if pct > 20 else "⭐" if pct > 10 else "·" if pct > 0 else "❌"
            print(f"  {label:35s} hit={r['hit_rate']:.2%} ({pct:+.0f}%) {mark}")

        print("=" * 65)
        return family_winners, all_main_results, plus_results


# ═══════════════════════════════════════════════════════════════════
# ANA SINIF
# ═══════════════════════════════════════════════════════════════════

class SansTopuPatternMasterV2:

    def __init__(self, path="sanstopu.xlsx", sheet="s1"):
        self.path, self.sheet = path, sheet
        self.df = None
        self._loader = None
        self.family_winners     = None
        self.all_main_results   = None
        self.plus_results       = None

    def load(self):
        self._loader = DataLoader(self.path, self.sheet)
        self.df = self._loader.load()
        print(f"✅ Veri yüklendi: {len(self.df)} çekiliş")
        print(f"📅 {self.df['tarih'].min():%d.%m.%Y} → {self.df['tarih'].max():%d.%m.%Y}")
        return self.df

    def get_main(self, row): return self._loader.main(row)
    def get_plus(self, row): return self._loader.plus(row)

    def run_tests(self, test_size=100, pool_size=10):
        eng = BacktestEngine(self.df, self.get_main, self.get_plus)
        self.family_winners, self.all_main_results, self.plus_results = \
            eng.run(test_size=test_size, pool_size=pool_size)
        return self.family_winners, self.all_main_results, self.plus_results

    def build_pool(self, pool_size=20):
        """
        Aile kazananlarından ağırlıklı oy ile pool oluşturur.
        pool_size=20 → P(5/5) ~%3.8 (17'den iyi)
        """
        if self.family_winners is None:
            raise RuntimeError("Önce run_tests() çalıştır")

        vote_scores = {n: 0.0 for n in range(1, 35)}
        total_weight = 0.0

        for family, (func_name, label, r, score) in self.family_winners.items():
            weight = max(0.01, score)  # negatif skoru düzelt
            p = MainPatterns(self.df, self.get_main)
            try:
                preds = getattr(p, func_name)(k=pool_size)
            except:
                continue
            for rank, num in enumerate(preds):
                vote_scores[num] += weight * (pool_size - rank)
            total_weight += weight

        ranked = sorted(vote_scores, key=vote_scores.get, reverse=True)
        return ranked[:pool_size]

    def best_plus_pool(self, k=5):
        """
        Artı için: hot + due karma skoru
        """
        if self.plus_results is None:
            raise RuntimeError("Önce run_tests() çalıştır")

        # Due skoru
        pp = PlusPatterns(self.df, self.get_plus)
        due = pp.due_score()   # {1..14: gün sayısı}
        hot_list = pp.hot(k=14)
        hot_rank = {n: i for i, n in enumerate(hot_list)}

        # Karma skor: 60% due ağırlığı + 40% hot sırası
        max_due = max(due.values()) or 1
        scores = {}
        for n in range(1, 15):
            due_norm = due[n] / max_due
            hot_norm = 1 - (hot_rank.get(n, 13) / 13)
            scores[n] = 0.6*due_norm + 0.4*hot_norm

        return sorted(scores, key=scores.get, reverse=True)[:k]

    def report(self, pool_size=20):
        pool     = self.build_pool(pool_size=pool_size)
        plus_pool = self.best_plus_pool(k=5)

        # Coverage simulation on last 50 draws
        hits50 = []
        for i in range(min(50, len(self.df))):
            actual = set(self.get_main(self.df.iloc[-(i+1)]))
            hits50.append(len(set(pool) & actual))
        coverage50 = sum(1 for i in range(min(50, len(self.df)))
                         if set(self.get_main(self.df.iloc[-(i+1)])).issubset(set(pool)))

        random_hits  = pool_size * 5 / 34
        actual_hits  = np.mean(hits50)
        improvement  = (actual_hits - random_hits) / random_hits * 100

        print("\n" + "=" * 65)
        print("🎯 ŞANS TOPU PATTERN MASTER v2 — RAPOR")
        print("=" * 65)
        print(f"📅 Son Çekiliş : {self.df['tarih'].max():%d.%m.%Y}")
        print(f"📊 Toplam      : {len(self.df)} çekiliş")

        print(f"\n{'─'*65}")
        print(f"🏆 ANA POOL ({pool_size} sayı)  ← Aile bazlı ağırlıklı ensemble")
        print(f"{'─'*65}")
        print(f"  {pool}")
        print(f"\n  Ortalama isabet (son 50): {actual_hits:.2f}/{pool_size} "
              f"(random={random_hits:.2f}, {improvement:+.1f}%)")
        print(f"  Coverage rate (son 50)  : {coverage50}/50 çekilişte 5/5 kapsandı "
              f"({coverage50/50:.1%})")

        print(f"\n{'─'*65}")
        print(f"🔥 ARTI POOL (due+hot karma skoru)")
        print(f"{'─'*65}")
        pp = PlusPatterns(self.df, self.get_plus)
        due = pp.due_score()
        for n in plus_pool:
            print(f"  {n:3d}  →  {due[n]:3d} çekilişdir çıkmadı")

        print(f"\n  Öneri: {plus_pool}")

        # Aile kazananları özeti
        print(f"\n{'─'*65}")
        print("📋 KAZANAN PATTERN AİLELERİ")
        print(f"{'─'*65}")
        for fam, (fn, lbl, r, score) in self.family_winners.items():
            print(f"  [{fam:12s}] {lbl:25s}  composite={score:.4f}  "
                  f"cov={r['coverage']:.1%}")

        print(f"\n{'─'*65}")
        print("⚠️  Eğlence amaçlıdır. Kazanç garantisi yoktur.")
        print("=" * 65)

        return {
            'main_pool':   pool,
            'plus_pool':   plus_pool,
            'avg_hits':    float(actual_hits),
            'coverage50':  int(coverage50),
            'improvement': float(improvement),
        }

    def save(self, result):
        os.makedirs('outputs', exist_ok=True)
        with open('outputs/sanstopu_pattern_master_v2.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print("\n💾 Kaydedildi: outputs/sanstopu_pattern_master_v2.json")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 65)
    print("🚀 ŞANS TOPU PATTERN MASTER v2")
    print("   Aile bazlı backtest · Coverage rate · Ağırlıklı ensemble")
    print("=" * 65)

    bot = SansTopuPatternMasterV2()
    bot.load()
    bot.run_tests(test_size=100, pool_size=10)
    result = bot.report(pool_size=20)
    bot.save(result)
    print("\n✅ TAMAMLANDI!")

if __name__ == "__main__":
    main()
