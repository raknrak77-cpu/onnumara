#!/usr/bin/env python3
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.prediction_engine import PredictionEngine

def run_basic():
    print("\n" + "="*50)
    print("🚀 On Numara Prediction Bot - Temel Analiz")
    print("="*50 + "\n")
    
    engine = PredictionEngine()
    engine.load_data()
    
    if len(engine.df) == 0:
        print("❌ Veri yüklenemedi! Excel dosyasını kontrol edin.")
        return None, None, None
    
    print(f"\n📊 Veri seti: {len(engine.df)} çekiliş")
    
    print("\n" + "-"*40)
    backtest_results = engine.run_backtest(train_size=500, test_size=50)
    
    print("\n" + "-"*40)
    print("🔮 İleri tahminler hesaplanıyor...")
    future_predictions = engine.predict_future(n_predictions=3)
    
    if future_predictions:
        print("\n🔮 Tahminler (Ensemble - Önerilen 10 Sayı):")
        for pred in future_predictions:
            print(f"\n   📅 {pred['tarih']}:")
            ensemble = pred['ensemble_top10']
            if ensemble:
                print(f"      {ensemble}")
            else:
                print("      (Modeller ortak sayı üretemedi)")
    
    engine.save_results(backtest_results, "backtest_results.json")
    engine.save_results(future_predictions, "forward_predictions.json")
    
    return engine, backtest_results, future_predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['basic', 'all'], default='all')
    args = parser.parse_args()
    
    if args.mode in ['basic', 'all']:
        run_basic()
    
    print("\n" + "="*50)
    print("✅ Tüm işlemler tamamlandı!")
    print("="*50)

if __name__ == "__main__":
    main()
