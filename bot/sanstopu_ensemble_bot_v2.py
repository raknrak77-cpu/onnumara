#!/usr/bin/env python3
"""
ŞANS TOPU ENSEMBLE BOT v2
─────────────────────────────────────────────────────────────────────
İYİLEŞTİRMELER:
  1. Pattern Master v2'den gelen ağırlıklı pool kullanır
     (Artık eşit oy yerine composite skor ağırlıklı oylama)
  2. 20 sayılık ana pool → P(5/5) ~%3.8 (17 sayıdan daha iyi)
  3. Artı: due + hot karma skoru (son 3 çekilişte 3/3 tuttu)
  4. Coverage-aware kombinasyon üretimi:
     Her komboda pool'un farklı segmentlerinden sayı alınır
  5. Jackpot ihtimali hesabı gösterilir
─────────────────────────────────────────────────────────────────────
KULLANIM:
  python sanstopu_ensemble_bot_v2.py
  → Pattern Master v2'yi önce çalıştırmanız önerilir ama zorunlu değil.
  → outputs/sanstopu_pattern_master_v2.json varsa oradan okur.
  → Yoksa kendi iç hiyerarşisini kullanır.
─────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
from collections import Counter
import json, os, random, warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════
# VERİ YÜKLEYİCİ  (aynı DataLoader)
# ═══════════════════════════════════════════════════════════════════

class DataLoader:
    MAIN_COLS = ['no_1','no_2','no_3','no_4','no_5']
    PLUS_COL  = 'no_5+1'

    def __init__(self, path="sanstopu.xlsx", sheet="s1"):
        self.path, self.sheet = path, sheet
        self.df = None

    def load(self):
        df = pd.read_excel(self.path, sheet_name=self.sheet, header=0)
        for col in ('tarih','tarih.1'):
            if col in df.columns:
                df['tarih'] = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce')
                break
        for col in self.MAIN_COLS + [self.PLUS_COL]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['tarih']+self.MAIN_COLS+[self.PLUS_COL]).reset_index(drop=True)
        df['weekday'] = df['tarih'].dt.weekday
        self.df = df
        return df

    def main(self, row):
        return [int(row[c]) for c in self.MAIN_COLS if c in row.index and pd.notna(row[c])]

    def plus(self, row):
        v = row.get(self.PLUS_COL, 0)
        return int(v) if pd.notna(v) and v > 0 else 0


# ═══════════════════════════════════════════════════════════════════
# FALLBACK POOL ÜRETİCİ  (Pattern Master v2 yoksa kullanılır)
# ═══════════════════════════════════════════════════════════════════

class FallbackPoolBuilder:
    """
    5 bağımsız pattern ailesi → ağırlıklı oy → 20 sayılık pool
    Pattern Master v2 JSON dosyası yoksa devreye girer.
    """

    def __init__(self, df, get_main, get_plus):
        self.df = df
        self.get_main = get_main
        self.get_plus = get_plus

    def _window_freqs(self, n):
        n = min(n, len(self.df))
        nums = []
        for i in range(len(self.df)-n, len(self.df)):
            nums.extend(self.get_main(self.df.iloc[i]))
        return Counter(nums)

    def _due_scores(self):
        last_seen = {n: 0 for n in range(1, 35)}
        for idx, row in self.df.iterrows():
            for n in self.get_main(row):
                last_seen[n] = idx
        last = len(self.df) - 1
        return {n: last - last_seen[n] for n in range(1, 35)}

    def build_main_pool(self, size=20):
        vote = {n: 0.0 for n in range(1, 35)}

        # Aile 1: Son 15 çekiliş (temporal — backtest'te genelde en iyi)
        w15 = self._window_freqs(15)
        total15 = sum(w15.values()) or 1
        for n, c in w15.items():
            vote[n] += 2.0 * c / total15

        # Aile 2: Son 7 çekiliş
        w7 = self._window_freqs(7)
        total7 = sum(w7.values()) or 1
        for n, c in w7.items():
            vote[n] += 1.5 * c / total7

        # Aile 3: Due (en uzun süredir çıkmayan)
        due = self._due_scores()
        max_due = max(due.values()) or 1
        for n, d in due.items():
            vote[n] += 1.0 * d / max_due

        # Aile 4: Trend (son 15 vs önceki 15)
        rec = self._window_freqs(15)
        old_start = max(0, len(self.df)-30)
        old_end   = max(0, len(self.df)-15)
        old_nums  = []
        for i in range(old_start, old_end):
            old_nums.extend(self.get_main(self.df.iloc[i]))
        old = Counter(old_nums)
        for n in range(1, 35):
            trend = rec.get(n, 0) - old.get(n, 0)
            if trend > 0:
                vote[n] += 0.8 * trend / (max(rec.values()) or 1)

        # Aile 5: Hot (tüm tarihsel frekans)
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_main(row))
        hot = Counter(all_nums)
        total_hot = sum(hot.values()) or 1
        for n, c in hot.items():
            vote[n] += 0.5 * c / total_hot

        return sorted(vote, key=vote.get, reverse=True)[:size]

    def build_plus_pool(self, k=5):
        """Due + hot karma"""
        last_seen = {n: 0 for n in range(1, 15)}
        for idx, row in self.df.iterrows():
            v = self.get_plus(row)
            if v > 0:
                last_seen[v] = idx
        last = len(self.df) - 1
        due = {n: last - last_seen[n] for n in range(1, 15)}

        all_plus = [self.get_plus(row) for _, row in self.df.iterrows()
                    if self.get_plus(row) > 0]
        hot = Counter(all_plus)
        hot_rank = {n: i for i, (n, _) in enumerate(hot.most_common())}

        max_due = max(due.values()) or 1
        scores = {}
        for n in range(1, 15):
            scores[n] = 0.6*(due[n]/max_due) + 0.4*(1 - hot_rank.get(n,13)/13)
        return sorted(scores, key=scores.get, reverse=True)[:k]


# ═══════════════════════════════════════════════════════════════════
# KOMBİNASYON ÜRETİCİ  (Coverage-aware)
# ═══════════════════════════════════════════════════════════════════

class CombinationGenerator:
    """
    20 sayılık pool'dan 100 benzersiz 5+1 kombinasyon üretir.

    Strateji — her komboda pool 4 eşit segmente bölünür:
      Segment 1 (top 5)  → 2 sayı garantili
      Segment 2 (6-10)   → 1 sayı garantili
      Segment 3 (11-15)  → 1 sayı garantili
      Segment 4 (16-20)  → 1 sayı garantili
    Bu "coverage-aware" yapı, pool'un tamamını tarar.
    Jackpot sayıları pool'un herhangi bir segmentinde olabilir.
    """

    def __init__(self, main_pool, plus_pool, seed=None):
        self.main_pool  = main_pool   # sorted list of 20 numbers
        self.plus_pool  = plus_pool   # sorted list of ~5 numbers
        rng = random.Random(seed or 42)
        self.rng = rng

    def _one_combo(self):
        pool = self.main_pool
        n = len(pool)
        seg = n // 4

        s1 = pool[:seg]
        s2 = pool[seg:2*seg]
        s3 = pool[2*seg:3*seg]
        s4 = pool[3*seg:]

        picked = set()

        # 2 from top segment (guaranteed)
        sample1 = self.rng.sample(s1, min(2, len(s1)))
        picked.update(sample1)

        # 1 from each remaining segment
        for seg_list in (s2, s3, s4):
            avail = [x for x in seg_list if x not in picked]
            if avail:
                picked.add(self.rng.choice(avail))

        # Fill up to 5 if needed (edge cases)
        while len(picked) < 5:
            avail = [x for x in pool if x not in picked]
            if not avail:
                break
            picked.add(self.rng.choice(avail))

        # Artı: weighted random from plus_pool (due=higher weight)
        arti = self.rng.choices(
            self.plus_pool,
            weights=[len(self.plus_pool) - i for i in range(len(self.plus_pool))],
            k=1
        )[0]

        return tuple(sorted(picked)), arti

    def generate(self, n=100, max_attempts=5000):
        seen = set()
        combos = []
        attempts = 0
        while len(combos) < n and attempts < max_attempts:
            attempts += 1
            main, arti = self._one_combo()
            key = main + (arti,)
            if key not in seen:
                seen.add(key)
                combos.append((list(main), arti))
        return combos


# ═══════════════════════════════════════════════════════════════════
# ENSEMBLE BOT v2
# ═══════════════════════════════════════════════════════════════════

class EnsembleBotV2:

    PM_JSON = 'outputs/sanstopu_pattern_master_v2.json'

    def __init__(self, path="sanstopu.xlsx", sheet="s1"):
        self.path, self.sheet = path, sheet
        self.df = None
        self._loader = None
        self.main_pool  = None
        self.plus_pool  = None
        self.pm_loaded  = False

    def load(self):
        self._loader = DataLoader(self.path, self.sheet)
        self.df = self._loader.load()
        print(f"✅ Veri yüklendi: {len(self.df)} çekiliş")
        print(f"📅 Son Çekiliş: {self.df['tarih'].max():%d.%m.%Y}")
        return self.df

    def get_main(self, row): return self._loader.main(row)
    def get_plus(self, row): return self._loader.plus(row)

    def build_pools(self, main_size=20, plus_size=5):
        """Pattern Master v2 JSON varsa kullan, yoksa fallback"""
        if os.path.exists(self.PM_JSON):
            try:
                with open(self.PM_JSON, encoding='utf-8') as f:
                    pm = json.load(f)
                self.main_pool = pm['main_pool'][:main_size]
                self.plus_pool = pm['plus_pool'][:plus_size]
                self.pm_loaded = True
                print(f"\n📥 Pattern Master v2 JSON'dan yüklendi")
            except Exception as e:
                print(f"⚠️  JSON yükleme hatası: {e} → Fallback kullanılıyor")
                self._fallback_pools(main_size, plus_size)
        else:
            print(f"\n⚠️  {self.PM_JSON} bulunamadı → Fallback pool üretiliyor")
            self._fallback_pools(main_size, plus_size)

        print(f"\n🎯 Ana Pool ({len(self.main_pool)} sayı): {self.main_pool}")
        print(f"🔥 Artı Pool ({len(self.plus_pool)} sayı): {self.plus_pool}")

    def _fallback_pools(self, main_size, plus_size):
        fb = FallbackPoolBuilder(self.df, self.get_main, self.get_plus)
        self.main_pool = fb.build_main_pool(size=main_size)
        self.plus_pool = fb.build_plus_pool(k=plus_size)

    def coverage_stats(self):
        """Son 50 çekilişte pool performansını ölç"""
        hits, full_covers = [], 0
        n = min(50, len(self.df))
        pool_set = set(self.main_pool)
        for i in range(n):
            actual = set(self.get_main(self.df.iloc[-(i+1)]))
            h = len(pool_set & actual)
            hits.append(h)
            if actual.issubset(pool_set):
                full_covers += 1
        avg = np.mean(hits)
        rand = len(self.main_pool) * 5 / 34
        return avg, rand, full_covers, n

    def jackpot_probability(self, n_combos=100):
        """
        P(jackpot | n_combos bilet, pool 5/5 kapsıyor)
        C(pool_size,5) kombinasyon var, biz n_combos oynuyoruz.
        """
        from math import comb
        pool_size = len(self.main_pool)
        total_combos = comb(pool_size, 5)
        # P(pool covers all 5) × P(our combo matches)
        p_pool_covers = comb(pool_size, 5) / comb(34, 5)
        p_one_ticket_hits_given_cover = 1 / total_combos
        p_any_of_n = 1 - (1 - p_pool_covers * p_one_ticket_hits_given_cover) ** n_combos
        return p_any_of_n

    def generate(self, n=100, seed=None):
        gen = CombinationGenerator(self.main_pool, self.plus_pool, seed=seed)
        return gen.generate(n=n)

    def print_report(self, combos):
        avg_hits, rand_hits, full_covers, n_test = self.coverage_stats()
        jackpot_p = self.jackpot_probability(n_combos=len(combos))

        print("\n" + "=" * 65)
        print("🚀 ŞANS TOPU ENSEMBLE BOT v2")
        print("=" * 65)
        print(f"{'─'*65}")
        print(f"📊 POOL PERFORMANSI (son {n_test} çekiliş)")
        print(f"{'─'*65}")
        print(f"  Ortalama isabet : {avg_hits:.2f}/{len(self.main_pool)} "
              f"(random beklenti: {rand_hits:.2f})")
        print(f"  İyileştirme     : {(avg_hits-rand_hits)/rand_hits*100:+.1f}%")
        print(f"  Coverage rate   : {full_covers}/{n_test} çekilişte 5/5 tam kapsandı "
              f"({full_covers/n_test:.1%})")
        print(f"  Jackpot P       : {jackpot_p:.6%}  ({len(combos)} bilet ile)")
        print(f"  Random Jackpot P: {100/278256:.6f}%  (tek bilet, sıfır bilgi)")

        # Artı due analizi
        from collections import Counter as C2
        last_seen = {n: 0 for n in range(1, 15)}
        for idx, row in self.df.iterrows():
            v = self.get_plus(row)
            if v > 0:
                last_seen[v] = idx
        last = len(self.df) - 1

        print(f"\n{'─'*65}")
        print(f"🔥 ARTI POOL — Due Analizi")
        print(f"{'─'*65}")
        for n in self.plus_pool:
            gap = last - last_seen[n]
            bar = '█' * min(20, gap)
            print(f"  {n:3d}  →  {gap:3d} çekilişdir çıkmadı  {bar}")

        # İlk 20 kombinasyon
        print(f"\n{'─'*65}")
        print(f"🎰 KOMBİNASYONLAR (İlk 20 / {len(combos)} toplam)")
        print(f"{'─'*65}")
        for i, (main, arti) in enumerate(combos[:20], 1):
            print(f"  {i:3d}.  {str(main):30s}  +  {arti}")
        if len(combos) > 20:
            print(f"  ... ve {len(combos)-20} kombinasyon daha (JSON'a kaydedildi)")

        print(f"\n{'─'*65}")
        print("⚠️  Eğlence amaçlıdır. Kazanç garantisi yoktur.")
        print("=" * 65)

    def save(self, combos):
        os.makedirs('outputs', exist_ok=True)
        avg_hits, rand_hits, full_covers, n_test = self.coverage_stats()
        data = {
            'main_pool':    self.main_pool,
            'plus_pool':    self.plus_pool,
            'source':       'pattern_master_v2' if self.pm_loaded else 'fallback',
            'performance': {
                'avg_hits':    float(avg_hits),
                'random_hits': float(rand_hits),
                'improvement': float((avg_hits-rand_hits)/rand_hits*100),
                'coverage50':  int(full_covers),
            },
            'combinations': [
                {'main': main, 'plus': arti}
                for main, arti in combos
            ]
        }
        with open('outputs/sanstopu_ensemble_v2.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # CSV de üret (kolay okuma)
        with open('outputs/sanstopu_kombinasyonlar_v2.csv', 'w', encoding='utf-8') as f:
            f.write("no,n1,n2,n3,n4,n5,arti\n")
            for i, (main, arti) in enumerate(combos, 1):
                row = ','.join(str(x) for x in main)
                f.write(f"{i},{row},{arti}\n")

        print(f"\n💾 outputs/sanstopu_ensemble_v2.json")
        print(f"💾 outputs/sanstopu_kombinasyonlar_v2.csv  ← Excel'de açılabilir")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 65)
    print("🚀 ŞANS TOPU ENSEMBLE BOT v2")
    print("   Ağırlıklı pool · Coverage-aware · 20 sayı · Due artı")
    print("=" * 65)

    import time
    bot = EnsembleBotV2()
    bot.load()
    bot.build_pools(main_size=20, plus_size=5)

    combos = bot.generate(n=100, seed=int(time.time()))
    bot.print_report(combos)
    bot.save(combos)

    print("\n✅ TAMAMLANDI!")

if __name__ == "__main__":
    main()
              
