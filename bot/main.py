#!/usr/bin/env python3
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.prediction_engine import PredictionEngine

def print_separator(char="=", length=60):
    print(char * length)

def run_advanced():
    print_separator()
    print("🚀 ON NUMARA GELİŞMİŞ TAHMİN BOTU (3.5+ HEDEFLİ)")
    print_separator()

    engine = PredictionEngine()
    engine.load_data()

    if len(engine.df) == 0:
        print("❌ Veri yüklenemedi!")
        return

    print(f"\n📅 Veri Aralığı: {engine.df['tarih'].min().strftime('%d.%m.%Y')} - {engine.df['tarih'].max().strftime('%d.%m.%Y')}")
    print(f"📊 Toplam Çekiliş: {len(engine.df)}")

    print("\n🔄 Backtest yapılıyor (son 50 çekiliş üzerinde)...")
    backtest_results = engine.run_backtest(train_size=500, test_size=50)

    print_separator("-")
    print("\n📈 YENİ MODEL BAŞARI SIRALAMASI")
    print_separator("-")

    sorted_models = sorted(backtest_results.keys(), key=lambda x: backtest_results[x]['avg_score'], reverse=True)

    for i, model in enumerate(sorted_models, 1):
        score = backtest_results[model]['avg_score']
        stars = "⭐" * int(score) + "☆" * (10 - int(score))
        print(f"\n{i}. {model:20} | {score:.2f}/10 {stars}")

    print_separator("-")
    print("\n🔮 GELECEK 3 ÇEKİLİŞ İÇİN TAHMİNLER")
    print_separator("-")

    future_preds = engine.predict_future_advanced(n_predictions=3)

    for pred in future_preds:
        print(f"\n📅 {pred['tarih']} TAHMİNLERİ:")
        print(f"   🎯 GELİŞMİŞ ENSEMBLE: {pred['advanced_ensemble']}")
        print(f"   🤖 ML RandomForest:   {pred['ml_prediction'][:5]}...")
        print(f"   📊 Ağırlıklı Frekans: {pred['weighted_freq'][:5]}...")
        print(f"   📈 Trend Analizi:     {pred['trend'][:5]}...")

    engine.save_results(backtest_results, "backtest_results_advanced.json")
    engine.save_results(future_preds, "forward_predictions_advanced.json")

    print_separator()
    print("✅ Tüm işlemler tamamlandı!")
    print_separator()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['basic', 'advanced'], default='advanced')
    args = parser.parse_args()

    if args.mode == 'advanced':
        run_advanced()
    else:
        print("Basit mod için --mode advanced kullanın")

if __name__ == "__main__":
    main()
