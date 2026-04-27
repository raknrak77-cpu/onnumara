#!/usr/bin/env python3
"""
On Numara Prediction Bot - Ana Modül
"""

import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.prediction_engine import PredictionEngine
from bot.advanced_models import AdvancedModels
from bot.visualizer import Visualizer

def run_basic():
    """Temel analizleri çalıştır"""
    print("\n" + "="*50)
    print("🚀 On Numara Prediction Bot - Temel Analiz")
    print("="*50 + "\n")
    
    engine = PredictionEngine()
    engine.load_data()
    
    print(f"📊 Veri seti: {len(engine.df)} çekiliş")
    
    # Backtest
    print("\n" + "-"*40)
    backtest_results = engine.run_backtest(train_size=500, test_size=50)
    
    # İleri tahmin
    print("\n" + "-"*40)
    print("🔮 İleri tahminler hesaplanıyor...")
    future_predictions = engine.predict_future(n_predictions=3)
    
    print("\n🔮 Tahminler (Ensemble - Önerilen 10 Sayı):")
    for pred in future_predictions:
        print(f"\n   📅 {pred['tarih']}:")
        print(f"      {pred['ensemble_top10']}")
    
    # Kaydet
    engine.save_results(backtest_results, "backtest_results.json")
    engine.save_results(future_predictions, "forward_predictions.json")
    
    return engine, backtest_results, future_predictions


def run_advanced(engine):
    """Gelişmiş analizleri çalıştır"""
    print("\n" + "="*50)
    print("🧠 On Numara Prediction Bot - Gelişmiş Analiz")
    print("="*50 + "\n")
    
    advanced = AdvancedModels(engine.df, engine.number_columns)
    
    # Apriori analizi
    print("📊 Apriori analizi yapılıyor...")
    apriori_results = advanced.apriori_analysis()
    if apriori_results:
        print("   En sık görülen sayı kümeleri:")
        for item in apriori_results[:5]:
            print(f"      {tuple(item['itemsets'])} -> destek: {item['support']:.3f}")
    
    # Zaman serisi trendleri
    print("\n📈 Zaman serisi analizi yapılıyor...")
    trends = advanced.time_series_analysis()
    hot_numbers = [num for num, t in trends.items() if t['trend'] == 'increasing'][:10]
    cold_numbers = [num for num, t in trends.items() if t['trend'] == 'decreasing'][:10]
    print(f"   🔥 Yükselişte: {hot_numbers[:5]}...")
    print(f"   ❄️ Düşüşte: {cold_numbers[:5]}...")
    
    # Ağırlıklı simülasyon
    print("\n🎲 Ağırlıklı simülasyon yapılıyor...")
    weighted_pred = advanced.weighted_simulation(n=10)
    print(f"   Ağırlıklı Monte Carlo: {weighted_pred}")
    
    # Random Forest
    print("\n🌲 Random Forest tahmini yapılıyor...")
    try:
        rf_pred = advanced.random_forest_prediction(n=10)
        print(f"   Random Forest: {rf_pred}")
    except Exception as e:
        print(f"   Random Forest hatası: {e}")
    
    return advanced


def run_visualize(engine, backtest_results, future_predictions):
    """Görselleştirme ve raporlama"""
    print("\n" + "="*50)
    print("📊 On Numara Prediction Bot - Görselleştirme")
    print("="*50 + "\n")
    
    viz = Visualizer(engine.df, engine.number_columns)
    viz.run_all(backtest_results, future_predictions)
    
    print("\n📁 Çıktılar:")
    print("   - outputs/plots/frequency_distribution.png")
    print("   - outputs/plots/number_trends.png")
    print("   - outputs/plots/cooccurrence_heatmap.png")
    print("   - reports/weekly_report.md")


def main():
    parser = argparse.ArgumentParser(description='On Numara Prediction Bot')
    parser.add_argument('--mode', choices=['basic', 'advanced', 'visualize', 'all'], 
                        default='all', help='Çalışma modu')
    args = parser.parse_args()
    
    if args.mode in ['basic', 'all']:
        engine, backtest_results, future_predictions = run_basic()
    
    if args.mode in ['advanced', 'all'] and 'engine' in locals():
        run_advanced(engine)
    elif args.mode in ['advanced', 'all']:
        engine, _, _ = run_basic()
        run_advanced(engine)
    
    if args.mode in ['visualize', 'all'] and 'engine' in locals():
        run_visualize(engine, backtest_results, future_predictions)
    elif args.mode in ['visualize', 'all']:
        engine, backtest_results, future_predictions = run_basic()
        run_visualize(engine, backtest_results, future_predictions)
    
    print("\n" + "="*50)
    print("✅ Tüm işlemler tamamlandı!")
    print("="*50)


if __name__ == "__main__":
    main()
