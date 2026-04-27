import argparse
from prediction_engine import PredictionEngine

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['backtest', 'forward', 'all'], default='all')
    args = parser.parse_args()
    
    engine = PredictionEngine(excel_path="onnumara_2020.xlsx")
    engine.load_and_clean()
    
    if args.mode in ['backtest', 'all']:
        print("\n🔙 Backtest yapılıyor...")
        backtest_results = engine.run_backtest(train_size=538, test_size=50)
        engine.save_results(backtest_results, "outputs/backtest.json")
        
    if args.mode in ['forward', 'all']:
        print("\n🔮 İleri tahmin yapılıyor...")
        forward_results = engine.predict_future(n_predictions=3)
        engine.save_results(forward_results, "outputs/forward_predictions.json")
        
if __name__ == "__main__":
    main()
