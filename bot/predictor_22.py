"""
22'lik Set Tahmin Botu
80 sayı içinden en güçlü 22 sayıyı önerir
"""

import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Dict
import json
import os
from datetime import timedelta

class Predictor22:
    def __init__(self, excel_path: str = "onnumara_2020.xlsx", sheet_name: str = "s1"):
        self.excel_path = excel_path
        self.sheet_name = sheet_name
        self.df = None
        self.number_columns = [f'no_{i}' for i in range(1, 23)]
        
    def load_data(self):
        self.df = pd.read_excel(self.excel_path, sheet_name=self.sheet_name, header=0)
        
        if 'tarih' in self.df.columns:
            self.df['tarih'] = pd.to_datetime(self.df['tarih'], format='%d.%m.%Y', errors='coerce')
        
        for col in self.number_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        self.df = self.df.dropna(subset=['tarih'] + self.number_columns, how='any')
        self.df = self.df.reset_index(drop=True)
        
        print(f"✅ Veri yüklendi: {len(self.df)} çekiliş")
        return self.df
    
    def get_valid_numbers(self, row) -> List[int]:
        nums = []
        for col in self.number_columns:
            if col in row.index and pd.notna(row[col]):
                try:
                    nums.append(int(row[col]))
                except:
                    pass
        return nums
    
    # ==================== MODEL 1: SON 5 ÇEKİLİŞTEKİ EN SIK 22 SAYI ====================
    def recent_22_model(self, n: int = 22) -> List[int]:
        """Son 5 çekilişte en sık çıkan 22 sayı"""
        recent_nums = []
        window = min(5, len(self.df))
        
        for idx in range(len(self.df) - window, len(self.df)):
            row = self.df.iloc[idx]
            recent_nums.extend(self.get_valid_numbers(row))
        
        counter = Counter(recent_nums)
        top_22 = [num for num, _ in counter.most_common(n)]
        
        # Eğer 22'den az varsa, frekans modelinden tamamla
        if len(top_22) < n:
            freq_pred = self.frequency_22_model(n)
            for num in freq_pred:
                if num not in top_22:
                    top_22.append(num)
                if len(top_22) == n:
                    break
        
        return top_22[:n]
    
    # ==================== MODEL 2: TÜM ZAMANLARIN EN SIK 22 SAYISI ====================
    def frequency_22_model(self, n: int = 22) -> List[int]:
        """Tüm zamanların en sık çıkan 22 sayısı"""
        all_nums = []
        for _, row in self.df.iterrows():
            all_nums.extend(self.get_valid_numbers(row))
        
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(n)]
    
    # ==================== MODEL 3: DUE NUMBERS (GECİKMİŞ) 22 ====================
    def due_22_model(self, n: int = 22) -> List[int]:
        """En uzun süredir çıkmayan 22 sayı"""
        last_seen = {num: 0 for num in range(1, 81)}
        
        for idx, row in self.df.iterrows():
            for num in self.get_valid_numbers(row):
                last_seen[num] = idx
        
        last_idx = len(self.df) - 1
        due_counts = {num: last_idx - last_seen[num] for num in range(1, 81)}
        
        due_sorted = sorted(due_counts.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in due_sorted[:n]]
    
    # ==================== MODEL 4: AĞIRLIKLI FREKANS (Zamana göre) ====================
    def weighted_frequency_22(self, n: int = 22, decay: float = 0.95) -> List[int]:
        """Yeni çekilişlere daha fazla ağırlık veren frekans"""
        weighted_counts = {}
        
        for idx, row in self.df.iterrows():
            weight = decay ** (len(self.df) - idx - 1)
            for num in self.get_valid_numbers(row):
                weighted_counts[num] = weighted_counts.get(num, 0) + weight
        
        sorted_nums = sorted(weighted_counts.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:n]]
    
    # ==================== MODEL 5: TREND ANALİZİ (Yükselen sayılar) ====================
    def trend_22_model(self, n: int = 22, window: int = 20) -> List[int]:
        """Son window çekilişte trendi artan sayılar"""
        scores = {}
        
        for num in range(1, 81):
            recent_count = 0
            older_count = 0
            
            recent_window = self.df.iloc[-window:]
            older_window = self.df.iloc[-window*2:-window] if len(self.df) > window*2 else self.df.iloc[:window]
            
            for _, row in recent_window.iterrows():
                if num in self.get_valid_numbers(row):
                    recent_count += 1
            
            for _, row in older_window.iterrows():
                if num in self.get_valid_numbers(row):
                    older_count += 1
            
            if older_count > 0:
                trend_score = (recent_count - older_count) / older_count
            else:
                trend_score = recent_count if recent_count > 0 else 0
            
            # Son görülme zamanı bonusu
            last_seen = 0
            for idx, row in self.df.iterrows():
                if num in self.get_valid_numbers(row):
                    last_seen = idx
            
            due = len(self.df) - last_seen - 1
            due_bonus = 1 / (due + 1) if due > 0 else 1
            
            scores[num] = trend_score * 10 + recent_count * 2 + due_bonus * 3
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:n]]
    
    # ==================== MODEL 6: FİZİKSEL 22 (Top ağırlığı, aşınma) ====================
    def physical_22_model(self, n: int = 22) -> List[int]:
        """Fiziksel faktörleri dikkate alan tahmin"""
        
        scores = {}
        
        # 1. Top yaşlanması - son çeyrekte artan sayılar
        quarter = len(self.df) // 4
        last_quarter = self.df.iloc[-quarter:]
        first_quarter = self.df.iloc[:quarter]
        
        last_counts = Counter()
        first_counts = Counter()
        
        for _, row in last_quarter.iterrows():
            last_counts.update(self.get_valid_numbers(row))
        for _, row in first_quarter.iterrows():
            first_counts.update(self.get_valid_numbers(row))
        
        for num in range(1, 81):
            last_freq = last_counts[num] / (quarter * 22) if quarter > 0 else 0
            first_freq = first_counts[num] / (quarter * 22) if quarter > 0 else 0
            aging_score = (last_freq - first_freq) * 20
            scores[num] = aging_score
        
        # 2. Kalibrasyon drift'i - son 20 çekilişte artanlar
        recent_counts = Counter()
        for _, row in self.df.iloc[-20:].iterrows():
            recent_counts.update(self.get_valid_numbers(row))
        
        for num in range(1, 81):
            recent_freq = recent_counts[num] / (20 * 22) if recent_counts[num] > 0 else 0.01
            scores[num] = scores.get(num, 0) + recent_freq * 10
        
        # 3. Kümelenme - hangi aralıklar yoğun?
        ranges = [(1, 20), (21, 40), (41, 60), (61, 80)]
        range_counts = {}
        for r in ranges:
            range_counts[r] = 0
        
        for _, row in self.df.iloc[-20:].iterrows():
            for num in self.get_valid_numbers(row):
                for r in ranges:
                    if r[0] <= num <= r[1]:
                        range_counts[r] += 1
                        break
        
        best_range = max(range_counts, key=range_counts.get)
        for num in range(best_range[0], best_range[1] + 1):
            scores[num] = scores.get(num, 0) + 5
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:n]]
    
    # ==================== ENSEMBLE 22 (Tüm modellerin ortak önerisi) ====================
    def ensemble_22_model(self, n: int = 22) -> Dict:
        """Tüm modelleri birleştir, kesişim ve ağırlıklı skor hesapla"""
        
        models = {
            'recent': self.recent_22_model(30),
            'frequency': self.frequency_22_model(30),
            'due': self.due_22_model(30),
            'weighted': self.weighted_frequency_22(30),
            'trend': self.trend_22_model(30),
            'physical': self.physical_22_model(30)
        }
        
        # Kesişim (tüm modellerde ortak olan sayılar)
        all_sets = [set(models[m]) for m in models]
        intersection = list(set.intersection(*all_sets))
        
        # Ağırlıklı skor
        weighted_scores = {}
        weights = {
            'recent': 1.5,
            'frequency': 1.0,
            'due': 0.8,
            'weighted': 1.2,
            'trend': 1.3,
            'physical': 1.4
        }
        
        for model_name, preds in models.items():
            weight = weights.get(model_name, 1.0)
            for rank, num in enumerate(preds):
                score = (30 - rank) * weight
                weighted_scores[num] = weighted_scores.get(num, 0) + score
        
        # Kesişimdekilere bonus
        for num in intersection:
            weighted_scores[num] = weighted_scores.get(num, 0) + 30
        
        sorted_scores = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
        final_22 = [num for num, _ in sorted_scores[:n]]
        
        return {
            'final_22': final_22,
            'intersection': intersection,
            'model_predictions': models,
            'scores': dict(sorted_scores[:30])
        }
    
    # ==================== RAPORLAMA ====================
    def generate_report(self) -> str:
        """22'lik set raporu oluştur"""
        
        ensemble = self.ensemble_22_model()
        
        report = []
        report.append("=" * 60)
        report.append("🎯 ON NUMARA 22'LİK SET TAHMİN RAPORU")
        report.append("=" * 60)
        report.append(f"\n📅 Son Çekiliş: {self.df['tarih'].max().strftime('%d.%m.%Y')}")
        report.append(f"📊 Toplam Analiz: {len(self.df)} çekiliş")
        
        report.append("\n" + "-" * 60)
        report.append("🏆 EN GÜÇLÜ 22 SAYI (Ensemble Model)")
        report.append("-" * 60)
        
        final_22 = ensemble['final_22']
        # 5'erli gruplar halinde göster
        for i in range(0, len(final_22), 5):
            group = final_22[i:i+5]
            report.append(f"  {i+1:2d}-{i+5:2d}. {' '.join(f'{n:3d}' for n in group)}")
        
        report.append("\n" + "-" * 60)
        report.append("🎯 KESİŞİM (Tüm modellerin ortak önerdiği sayılar)")
        report.append("-" * 60)
        
        intersection = ensemble['intersection']
        if intersection:
            report.append(f"  {len(intersection)} adet: {intersection}")
        else:
            report.append("  (Ortak sayı bulunamadı - beklenen)")
        
        report.append("\n" + "-" * 60)
        report.append("📊 MODELLERİN İLK 10 TAHMİNİ")
        report.append("-" * 60)
        
        for model_name, preds in ensemble['model_predictions'].items():
            report.append(f"\n  {model_name.upper()}:")
            report.append(f"    {preds[:10]}")
        
        report.append("\n" + "-" * 60)
        report.append("⚠️ NOT: Bu 22 sayı, geçmiş verilere dayalı istatistiksel analizdir.")
        report.append("   Kesin sonuç garantisi yoktur. Eğlence amaçlıdır.")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def save_results(self, results: Dict, filename: str):
        os.makedirs('outputs', exist_ok=True)
        with open(f'outputs/{filename}', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"💾 Kaydedildi: outputs/{filename}")
