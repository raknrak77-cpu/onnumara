"""
Gelişmiş Tahmin Motoru - 3.5/10 başarı hedefli
"""

import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Dict
import json
from datetime import timedelta
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d')
        return super().default(obj)


class PredictionEngine:
    def __init__(self, excel_path: str = "onnumara_2020.xlsx", sheet_name: str = "s1"):
        self.excel_path = excel_path
        self.sheet_name = sheet_name
        self.df = None
        self.number_columns = [f'no_{i}' for i in range(1, 23)]
        self.scaler = StandardScaler()
        self.rf_model = None
        self.gb_model = None
        
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
    
    # ==================== YENİ MODEL 1: AĞIRLIKLI FREKANS (Zamana göre) ====================
    def weighted_frequency_prediction(self, train_df: pd.DataFrame, n: int = 10, decay: float = 0.95) -> List[int]:
        """Yeni çekilişlere daha fazla ağırlık veren frekans analizi"""
        all_nums_with_weights = []
        
        for idx, row in train_df.iterrows():
            # Ağırlık: son çekilişe yaklaştıkça artar
            weight = decay ** (len(train_df) - idx - 1)
            nums = self.get_valid_numbers(row)
            for num in nums:
                all_nums_with_weights.append((num, weight))
        
        # Ağırlıklı frekans hesapla
        weighted_counts = {}
        for num, weight in all_nums_with_weights:
            weighted_counts[num] = weighted_counts.get(num, 0) + weight
        
        sorted_nums = sorted(weighted_counts.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:n]]
    
    # ==================== YENİ MODEL 2: HAREKETLİ ORTALAMA + TREND ====================
    def trend_prediction(self, train_df: pd.DataFrame, n: int = 10, window: int = 10) -> List[int]:
        """Her sayının son window çekilişteki trendini analiz eder"""
        trends = {}
        
        for num in range(1, 81):
            # Son window çekilişteki görülme sıklığı
            recent_count = 0
            older_count = 0
            
            recent_window = train_df.iloc[-window:]
            older_window = train_df.iloc[-window*2:-window] if len(train_df) > window*2 else train_df.iloc[:window]
            
            for _, row in recent_window.iterrows():
                if num in self.get_valid_numbers(row):
                    recent_count += 1
            
            for _, row in older_window.iterrows():
                if num in self.get_valid_numbers(row):
                    older_count += 1
            
            # Trend: artıyorsa pozitif, azalıyorsa negatif
            if older_count > 0:
                trend = (recent_count - older_count) / older_count
            else:
                trend = recent_count if recent_count > 0 else 0
            
            # Son görülme zamanı (due)
            last_seen = 0
            for idx, row in train_df.iterrows():
                if num in self.get_valid_numbers(row):
                    last_seen = idx
            
            due = len(train_df) - last_seen - 1
            
            # Kombine skor (trend + recent_count + due faktörü)
            score = recent_count * 2 + trend * 5 + (1 / (due + 1)) * 3
            trends[num] = score
        
        sorted_nums = sorted(trends.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:n]]
    
    # ==================== YENİ MODEL 3: ISI HARİTASI + BİRLİKTELİK ====================
    def cooccurrence_enhanced(self, train_df: pd.DataFrame, n: int = 10, top_k: int = 5) -> List[int]:
        """Son çekilişteki sayılarla birlikte çıkma olasılığı yüksek olanları bulur"""
        # Son çekilişteki sayılar
        last_row = train_df.iloc[-1]
        last_numbers = set(self.get_valid_numbers(last_row))
        
        if not last_numbers:
            return self.weighted_frequency_prediction(train_df, n)
        
        # Birliktelik matrisi oluştur
        cooccurrence = {}
        for num in range(1, 81):
            cooccurrence[num] = {}
            for other in range(1, 81):
                cooccurrence[num][other] = 0
        
        for _, row in train_df.iterrows():
            nums = self.get_valid_numbers(row)
            for i, n1 in enumerate(nums):
                for n2 in nums[i+1:]:
                    cooccurrence[n1][n2] += 1
                    cooccurrence[n2][n1] += 1
        
        # Son sayılarla en çok birlikte çıkanları bul
        candidate_scores = {}
        for num in range(1, 81):
            if num in last_numbers:
                continue
            score = 0
            for last_num in last_numbers:
                score += cooccurrence[num].get(last_num, 0)
            candidate_scores[num] = score
        
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_candidates[:n]]
    
    # ==================== YENİ MODEL 4: RANDOM FOREST (Makine Öğrenmesi) ====================
    def train_ml_models(self, train_df: pd.DataFrame):
        """Random Forest ve Gradient Boosting modellerini eğit"""
        X = []
        y = []
        
        # Özellik mühendisliği: son 3 çekilişin özet istatistikleri
        for idx in range(3, len(train_df)):
            features = []
            
            # Son 3 çekilişin tüm sayıları
            recent_numbers = []
            for j in range(1, 4):
                row = train_df.iloc[idx - j]
                recent_numbers.extend(self.get_valid_numbers(row))
            
            # Özellikler
            features.append(len(set(recent_numbers)))  # Benzersiz sayı sayısı
            features.append(np.mean(recent_numbers))   # Ortalama
            features.append(np.std(recent_numbers))    # Standart sapma
            
            # Sayı aralıklarına göre dağılım
            ranges = [1-20, 21-40, 41-60, 61-80]
            for r in ranges:
                count = sum(1 for x in recent_numbers if r[0] <= x <= r[1])
                features.append(count)
            
            # Tek/çift sayı oranı
            odd_count = sum(1 for x in recent_numbers if x % 2 == 1)
            features.append(odd_count / 22)
            
            X.append(features)
            y.extend(self.get_valid_numbers(train_df.iloc[idx]))
        
        if len(X) > 100:
            X = np.array(X)
            self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            self.gb_model = GradientBoostingClassifier(n_estimators=50, random_state=42)
            
            # Her sayı için ayrı model yerine, tüm sayıları eğit
            self.rf_model.fit(X, y[:len(X)])
            self.gb_model.fit(X, y[:len(X)])
    
    def ml_prediction(self, train_df: pd.DataFrame, n: int = 10) -> List[int]:
        """ML modeli ile tahmin"""
        if self.rf_model is None:
            self.train_ml_models(train_df)
        
        if self.rf_model is None:
            return self.weighted_frequency_prediction(train_df, n)
        
        # Son 3 çekiliş için özellik çıkar
        recent_numbers = []
        for j in range(1, 4):
            row = train_df.iloc[-j]
            recent_numbers.extend(self.get_valid_numbers(row))
        
        features = []
        features.append(len(set(recent_numbers)))
        features.append(np.mean(recent_numbers))
        features.append(np.std(recent_numbers))
        
        ranges = [(1, 20), (21, 40), (41, 60), (61, 80)]
        for r in ranges:
            count = sum(1 for x in recent_numbers if r[0] <= x <= r[1])
            features.append(count)
        
        odd_count = sum(1 for x in recent_numbers if x % 2 == 1)
        features.append(odd_count / 22)
        
        # Tahmin olasılıkları
        X_pred = np.array([features])
        
        try:
            rf_proba = self.rf_model.predict_proba(X_pred)[0]
            gb_proba = self.gb_model.predict_proba(X_pred)[0]
            
            # Ortalama olasılık
            avg_proba = (rf_proba + gb_proba) / 2
            
            # En yüksek olasılıklı n sayı
            top_indices = np.argsort(avg_proba)[-n:][::-1]
            return [int(self.rf_model.classes_[i]) for i in top_indices]
        except:
            return self.weighted_frequency_prediction(train_df, n)
    
    # ==================== YENİ MODEL 5: ÇOKLU MARKOV (Son 3 sayı) ====================
    def multi_markov_prediction(self, train_df: pd.DataFrame, n: int = 10) -> List[int]:
        """Son 3 sayıya göre geçiş olasılıkları"""
        if len(train_df) < 3:
            return self.frequency_prediction(train_df, n)
        
        # Üçlü geçişler (trigram)
        transitions = {}
        
        for _, row in train_df.iterrows():
            nums = self.get_valid_numbers(row)
            for i in range(len(nums)-2):
                key = (nums[i], nums[i+1])
                next_num = nums[i+2]
                if key not in transitions:
                    transitions[key] = []
                transitions[key].append(next_num)
        
        # Son çekilişin son 2 sayısı
        last_row = train_df.iloc[-1]
        last_nums = self.get_valid_numbers(last_row)
        
        if len(last_nums) >= 2:
            key = (last_nums[-2], last_nums[-1])
            if key in transitions:
                counter = Counter(transitions[key])
                return [num for num, _ in counter.most_common(n)]
        
        return self.weighted_frequency_prediction(train_df, n)
    
    # ==================== YENİ MODEL 6: DÖNGÜSEL ANALİZ (Haftalık periyot) ====================
    def cyclical_prediction(self, train_df: pd.DataFrame, n: int = 10) -> List[int]:
        """Haftanın gününe göre benzer geçmiş çekilişleri bulur"""
        if 'tarih' not in train_df.columns:
            return self.weighted_frequency_prediction(train_df, n)
        
        # Son çekilişin haftanın günü
        last_date = train_df['tarih'].iloc[-1]
        last_weekday = last_date.weekday()  # 0=Pazartesi, 4=Cuma
        
        # Aynı hafta günündeki çekilişleri bul
        similar_draws = []
        for idx, row in train_df.iterrows():
            if row['tarih'].weekday() == last_weekday:
                similar_draws.append(idx)
        
        if len(similar_draws) < 5:
            return self.weighted_frequency_prediction(train_df, n)
        
        # Benzer çekilişlerden en sık çıkan sayılar
        similar_numbers = []
        for idx in similar_draws[-20:]:  # Son 20 benzer çekiliş
            row = train_df.iloc[idx]
            similar_numbers.extend(self.get_valid_numbers(row))
        
        counter = Counter(similar_numbers)
        return [num for num, _ in counter.most_common(n)]
    
    # ==================== ENSEMBLE (Gelişmiş) ====================
    def advanced_ensemble(self, train_df: pd.DataFrame, n: int = 10) -> List[int]:
        """Tüm gelişmiş modellerin ağırlıklı oylaması"""
        models = {
            'weighted_freq': self.weighted_frequency_prediction,
            'trend': self.trend_prediction,
            'cooccurrence': self.cooccurrence_enhanced,
            'multi_markov': self.multi_markov_prediction,
            'cyclical': self.cyclical_prediction
        }
        
        # Her modelden tahmin al
        all_votes = []
        model_weights = {
            'weighted_freq': 1.0,
            'trend': 1.2,      # Daha başarılı bekleniyor
            'cooccurrence': 1.1,
            'multi_markov': 0.9,
            'cyclical': 0.8
        }
        
        for model_name, model_func in models.items():
            preds = model_func(train_df, n * 2)
            weight = model_weights.get(model_name, 1.0)
            for num in preds:
                all_votes.extend([num] * int(weight * 2))
        
        counter = Counter(all_votes)
        return [num for num, _ in counter.most_common(n)]
    
    # ==================== BACKTEST (Tüm modeller) ====================
    def run_backtest(self, train_size: int = 500, test_size: int = 50) -> Dict:
        if self.df is None:
            self.load_data()
        
        total = len(self.df)
        train_size = min(train_size, total - test_size)
        
        if train_size <= 0:
            return {}
        
        models = {
            'recent_old': self.recent_prediction,
            'frequency_old': self.frequency_prediction,
            'due_old': self.due_numbers_prediction,
            'weighted_freq': self.weighted_frequency_prediction,
            'trend': self.trend_prediction,
            'cooccurrence': self.cooccurrence_enhanced,
            'multi_markov': self.multi_markov_prediction,
            'cyclical': self.cyclical_prediction,
            'advanced_ensemble': self.advanced_ensemble
        }
        
        results = {name: {'scores': [], 'total_correct': 0, 'total_tested': 0} for name in models}
        
        print(f"📊 Backtest: {train_size} eğitim, {test_size} test")
        
        for i in range(min(test_size, total - train_size)):
            train_end = train_size + i
            train_df = self.df.iloc[:train_end]
            test_row = self.df.iloc[train_end]
            actual = set(self.get_valid_numbers(test_row))
            
            # ML modelini her 10 testte bir yeniden eğit
            if i % 10 == 0:
                self.train_ml_models(train_df)
            
            # ML tahmini
            ml_pred = self.ml_prediction(train_df, 10)
            results['ml_randomforest'] = results.get('ml_randomforest', {'scores': [], 'total_correct': 0, 'total_tested': 0})
            correct = len(set(ml_pred) & actual)
            results['ml_randomforest']['scores'].append(correct)
            results['ml_randomforest']['total_correct'] += correct
            results['ml_randomforest']['total_tested'] += 1
            
            # Diğer modeller
            for name, func in models.items():
                preds = func(train_df, 10)
                correct = len(set(preds) & actual)
                results[name]['scores'].append(correct)
                results[name]['total_correct'] += correct
                results[name]['total_tested'] += 1
        
        # Ortalamaları hesapla
        for name in results:
            if results[name]['total_tested'] > 0:
                results[name]['avg_score'] = results[name]['total_correct'] / results[name]['total_tested']
            else:
                results[name]['avg_score'] = 0
        
        return results
    
    # ==================== MEVCUT MODELLER (Korunan) ====================
    def frequency_prediction(self, train_df: pd.DataFrame, n: int = 10) -> List[int]:
        all_nums = []
        for col in self.number_columns:
            if col in train_df.columns:
                all_nums.extend([int(x) for x in train_df[col].dropna().tolist()])
        if not all_nums:
            return list(range(1, n+1))
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(n)]
    
    def due_numbers_prediction(self, train_df: pd.DataFrame, n: int = 10) -> List[int]:
        last_seen = {num: 0 for num in range(1, 81)}
        for idx, row in train_df.iterrows():
            nums = self.get_valid_numbers(row)
            for num in nums:
                last_seen[num] = idx
        last_idx = len(train_df) - 1 if len(train_df) > 0 else 0
        due_counts = {num: last_idx - last_seen[num] for num in range(1, 81)}
        due_sorted = sorted(due_counts.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in due_sorted[:n]]
    
    def recent_prediction(self, train_df: pd.DataFrame, n: int = 10, window: int = 5) -> List[int]:
        recent_nums = []
        start_idx = max(0, len(train_df) - window)
        for idx in range(start_idx, len(train_df)):
            row_nums = self.get_valid_numbers(train_df.iloc[idx])
            recent_nums.extend(row_nums)
        counter = Counter(recent_nums)
        return [num for num, _ in counter.most_common(n)]
    
    def markov_prediction(self, train_df: pd.DataFrame, n: int = 10) -> List[int]:
        if len(train_df) == 0:
            return self.frequency_prediction(train_df, n)
        transitions = {}
        for _, row in train_df.iterrows():
            nums = self.get_valid_numbers(row)
            for i in range(len(nums)-1):
                current, next_num = nums[i], nums[i+1]
                if current not in transitions:
                    transitions[current] = []
                transitions[current].append(next_num)
        last_row = train_df.iloc[-1]
        last_nums = self.get_valid_numbers(last_row)
        if not last_nums:
            return self.frequency_prediction(train_df, n)
        last_num = last_nums[-1]
        if last_num in transitions:
            counter = Counter(transitions[last_num])
            return [num for num, _ in counter.most_common(n)]
        return self.frequency_prediction(train_df, n)
    
def monte_carlo_prediction(self, train_df: pd.DataFrame, n: int = 10, simulations: int = 5000) -> List[int]:
        all_nums = []
        for col in self.number_columns:
            if col in train_df.columns:
                all_nums.extend([int(x) for x in train_df[col].dropna().tolist()])
        if not all_nums:
            return list(range(1, n+1))
        probs = {}
        total = len(all_nums)
        for num in range(1, 81):
            count = all_nums.count(num)
            probs[num] = (count + 1) / (total + 80)
        simulated_counts = Counter()
        for _ in range(simulations):
            selected = np.random.choice(list(probs.keys()), size=22, replace=False, p=list(probs.values()))
            simulated_counts.update(selected)
        return [num for num, _ in simulated_counts.most_common(n)]
    
    def predict_future_advanced(self, n_predictions: int = 3) -> List[Dict]:
        """Gelişmiş gelecek tahminleri - her çekiliş için farklı"""
        if self.df is None:
            self.load_data()
        
        if len(self.df) == 0:
            return []
        
        future_predictions = []
        last_date = self.df['tarih'].max()
        
        # Her çekiliş için ayrı ayrı tahmin yap
        current_df = self.df.copy()
        
        for i in range(n_predictions):
            next_date = last_date + timedelta(days=3 + i*4)
            
            # Mevcut veri ile tahmin
            pred = {
                'tahmin_no': i + 1,
                'tarih': next_date.strftime('%d.%m.%Y'),
                'weighted_freq': self.weighted_frequency_prediction(current_df, 10),
                'trend': self.trend_prediction(current_df, 10),
                'cooccurrence': self.cooccurrence_enhanced(current_df, 10),
                'multi_markov': self.multi_markov_prediction(current_df, 10),
                'cyclical': self.cyclical_prediction(current_df, 10),
                'advanced_ensemble': self.advanced_ensemble(current_df, 10)
            }
            future_predictions.append(pred)
        
        return future_predictions
    
    def save_results(self, results, filename: str):
        os.makedirs('outputs', exist_ok=True)
        with open(f'outputs/{filename}', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        print(f"💾 Kaydedildi: outputs/{filename}")
