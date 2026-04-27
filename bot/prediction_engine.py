"""
Tahmin motoru - Backtest ve ileri tahmin yapar
"""

import pandas as pd
import numpy as np
from collections import Counter
import json
from datetime import timedelta

class PredictionEngine:
    def __init__(self, excel_path="onnumara_2020.xlsx"):
        self.excel_path = excel_path
        self.df = None
        self.number_columns = []  # Dinamik olarak bulunacak
        
    def load_and_clean(self):
        """Excel'den veri yükle ve temizle - Sayfa7 için özel"""
        self.df = pd.read_excel(self.excel_path, sheet_name="Sayfa7", header=3)
        
        print(f"📋 İlk satır: {self.df.iloc[0].tolist()[:5]}...")
        print(f"📋 Sütunlar: {list(self.df.columns[:10])}...")
        
        # Sütun isimlerini düzenle
        self.df.columns = self.df.columns.astype(str).str.strip()
        
        # Sayı sütunlarını bul (içinde sayı olan, float veya int değerler)
        number_cols = []
        for col in self.df.columns:
            # Sayı sütunları genelde 1,2,3.. veya no-1, no1 gibi
            col_str = str(col).lower()
            # İlk birkaç satırdaki değerlere bak
            sample = self.df[col].dropna().iloc[:5] if len(self.df[col].dropna()) > 0 else []
            if len(sample) > 0:
                # Değerler 1-80 arasında sayıysa bu bir sayı sütunu
                try:
                    numeric_sample = pd.to_numeric(sample, errors='coerce')
                    if numeric_sample.notna().all() and all(1 <= x <= 80 for x in numeric_sample if pd.notna(x)):
                        number_cols.append(col)
                except:
                    pass
        
        print(f"🔢 Bulunan sayı sütunları: {number_cols[:5]}... (toplam {len(number_cols)})")
        
        # Eğer otomatik bulamadıysa, B'den W'ye kadar olan sütunları dene
        if len(number_cols) < 22:
            possible_cols = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
                           'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W']
            # A harfinden başlayan sütunlar için
            for i, letter in enumerate(possible_cols, 1):
                if letter in self.df.columns:
                    number_cols.append(letter)
        
        self.number_columns = number_cols[:22]  # En fazla 22 sütun
        
        # Tarih sütununu bul
        date_col = None
        for col in self.df.columns:
            col_str = str(col).lower()
            sample = str(self.df[col].iloc[0]) if len(self.df) > 0 else ""
            if 'çekiliş' in col_str or 'tarih' in col_str or ('/' in sample and len(sample) == 10):
                date_col = col
                break
        
        # Orijinal tarih sütununu koruyarak yeni bir sütun oluştur
        if date_col:
            self.df['tarih_raw'] = self.df[date_col].astype(str)
            # "1.Çekiliş[1] 03/08/2020" formatındaysa ayır
            self.df['tarih_str'] = self.df['tarih_raw'].str.split().str[-1]
            self.df['tarih'] = pd.to_datetime(self.df['tarih_str'], format='%d/%m/%Y', errors='coerce')
            print(f"📅 Tarih sütunu: {date_col}, örnek: {self.df['tarih'].iloc[0] if len(self.df) > 0 else 'yok'}")
        
        # Sayı sütunlarını sayısal yap
        for col in self.number_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # NaN satırları temizle
        if 'tarih' in self.df.columns:
            self.df = self.df.dropna(subset=['tarih'])
        if self.number_columns:
            self.df = self.df.dropna(subset=self.number_columns, how='any')
        
        # Çekiliş numarası ekle
        self.df['no'] = range(1, len(self.df) + 1)
        
        print(f"✅ Yüklendi: {len(self.df)} çekiliş")
        print(f"📊 Sütunlar: {list(self.df.columns)}")
        return self.df
    
    def get_all_numbers_series(self):
        """Tüm sayıları tek liste olarak döndür"""
        if self.df is None:
            self.load_and_clean()
        if not self.number_columns:
            return []
        all_nums = []
        for col in self.number_columns:
            if col in self.df.columns:
                all_nums.extend(self.df[col].tolist())
        return all_nums
    
    def estimate_next_date(self, offset=1):
        """Bir sonraki çekiliş tarihini tahmin et"""
        if self.df is None:
            self.load_and_clean()
        if 'tarih' not in self.df.columns:
            return "Bilinmiyor"
        last_date = self.df['tarih'].max()
        next_date = last_date + timedelta(days=3 + (offset-1)*4)
        return next_date.strftime('%d.%m.%Y')
    
    def run_backtest(self, train_size=500, test_size=50):
        """Geriye dönük test"""
        if self.df is None:
            self.load_and_clean()
        
        if len(self.df) < train_size + test_size:
            print(f"⚠️ Veri boyutu yetersiz: {len(self.df)} çekiliş var, {train_size+test_size} gerekli")
            train_size = len(self.df) - test_size
        
        results = []
        
        for i in range(test_size):
            train_end = train_size + i
            if train_end >= len(self.df):
                break
                
            train_df = self.df.iloc[:train_end]
            test_row = self.df.iloc[train_end]
            actual_numbers = [test_row[col] for col in self.number_columns if col in self.df.columns]
            
            if len(actual_numbers) < 22:
                continue
            
            predictions = {
                'frequency_based': self.frequency_prediction(train_df, 10),
                'markov_based': self.markov_prediction(train_df, 10),
                'monte_carlo': self.monte_carlo_prediction(train_df, 10),
                'random': self.random_prediction(10)
            }
            
            # Ensemble: 3 modelin ortak önerdiği sayılar
            all_pred_sets = [set(p) for p in [predictions['frequency_based'], 
                                               predictions['markov_based'], 
                                               predictions['monte_carlo']]]
            predictions['ensemble'] = list(set.intersection(*all_pred_sets)) if all_pred_sets else []
            
            score = len(set(predictions['ensemble']) & set(actual_numbers)) if predictions['ensemble'] else 0
            
            results.append({
                'cekilis_no': train_end + 1,
                'tarih': str(test_row['tarih'].date()) if 'tarih' in test_row else str(train_end + 1),
                'gercekler': actual_numbers,
                'tahminler': predictions,
                'basarisayisi': score
            })
        
        return results
    
    def predict_future(self, n_predictions=3):
        """Gelecek çekilişler için tahmin"""
        if self.df is None:
            self.load_and_clean()
            
        predictions = []
        for i in range(n_predictions):
            pred = {
                'tahmin_no': i+1,
                'tarih_tahmini': self.estimate_next_date(i+1),
                'frequency_top10': self.frequency_prediction(self.df, 10),
                'markov_top10': self.markov_prediction(self.df, 10),
                'monte_carlo_top10': self.monte_carlo_prediction(self.df, 10),
                'random_top10': self.random_prediction(10)
            }
            all_sets = [set(pred['frequency_top10']), set(pred['markov_top10']), set(pred['monte_carlo_top10'])]
            pred['ensemble_top10'] = list(set.intersection(*all_sets)) if all_sets else []
            predictions.append(pred)
            
        return predictions
    
    # ========== MODEL 1: Frekans bazlı ==========
    def frequency_prediction(self, train_df, n=10):
        """En sık çıkan n sayıyı döndür"""
        if not self.number_columns:
            return list(range(1, 11))
        
        all_nums = []
        for col in self.number_columns:
            if col in train_df.columns:
                all_nums.extend(train_df[col].dropna().tolist())
        
        if not all_nums:
            return list(range(1, 11))
            
        counter = Counter(all_nums)
        return [num for num, _ in counter.most_common(n)]
    
    # ========== MODEL 2: Markov zinciri ==========
    def markov_prediction(self, train_df, n=10):
        """Bir sayıdan sonra hangi sayı gelme ihtimali yüksek?"""
        if not self.number_columns:
            return self.frequency_prediction(train_df, n)
            
        transitions = {}
        
        for _, row in train_df.iterrows():
            nums = [row[col] for col in self.number_columns if col in row and pd.notna(row[col])]
            for j in range(len(nums)-1):
                current = nums[j]
                next_num = nums[j+1]
                if current not in transitions:
                    transitions[current] = []
                transitions[current].append(next_num)
        
        last_row = train_df.iloc[-1]
        last_nums = [last_row[col] for col in self.number_columns if col in last_row and pd.notna(last_row[col])]
        if not last_nums:
            return self.frequency_prediction(train_df, n)
            
        last_num = last_nums[-1]
        
        if last_num in transitions:
            next_nums = transitions[last_num]
            counter = Counter(next_nums)
            return [num for num, _ in counter.most_common(n)]
        return self.frequency_prediction(train_df, n)
    
    # ========== MODEL 3: Monte Carlo ==========
    def monte_carlo_prediction(self, train_df, n=10, simulations=5000):
        """Geçmiş dağılıma göre simülasyon"""
        if not self.number_columns:
            return list(range(1, 11))
            
        all_nums = []
        for col in self.number_columns:
            if col in train_df.columns:
                all_nums.extend(train_df[col].dropna().tolist())
        
        if not all_nums:
            return list(range(1, 11))
        
        probs = {}
        for num in range(1, 81):
            probs[num] = (all_nums.count(num) + 1) / (len(all_nums) + 80)
        
        simulated_counts = Counter()
        for _ in range(simulations):
            selected = np.random.choice(
                list(probs.keys()), 
                size=22, 
                replace=False, 
                p=list(probs.values())
            )
            simulated_counts.update(selected)
        
        return [num for num, _ in simulated_counts.most_common(n)]
    
    # ========== MODEL 4: Rastgele ==========
    def random_prediction(self, n=10):
        """Tamamen rastgele n sayı seç"""
        return sorted(np.random.choice(range(1, 81), n, replace=False).tolist())
    
    def save_results(self, results, filename):
        """Sonuçları JSON'a kaydet"""
        import os
        os.makedirs('outputs', exist_ok=True)
        with open(f'outputs/{filename}', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"💾 Kaydedildi: outputs/{filename}")
