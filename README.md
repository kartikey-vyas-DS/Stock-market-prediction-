# Stock market prediction 
 Stock market prediction using classification models 
![logo](logo.png)

Launch the web app:

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://indexesstockmarketml.streamlit.app/)


[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/Kvs8999/market-momentum-predictor)

# 📈 Market Momentum Predictor 📉

Welcome to the Market Momentum Predictor! This advanced machine learning-powered tool helps you forecast market trends for Indian stock indices.

## 🌟 Features

- 🤖 Utilizes four ML models: Random Forest, KNN, Decision Tree, and XGBoost
- 🔮 Predicts market momentum for 1-7 days in the future
- 📊 Provides individual predictions from each model with backtest accuracies
- 🧠 Uses a sophisticated ensemble method for final predictions
- 📈 Displays historical trend visualization

## 🚀 Quick Start

1. Visit our [Market Momentum Predictor App](https://huggingface.co/spaces/YourUsername/market-momentum-predictor)
2. Select an Indian stock ticker from the dropdown
3. Enter a prediction date (1-7 days in the future)
4. Click "Submit" and get your prediction!

## 💻 Tech Stack

- Python 3.10
- Gradio for the web interface
- Pandas & Numpy for data manipulation
- Scikit-learn, XGBoost for machine learning
- Matplotlib & Seaborn for visualization
- YFinance for fetching stock data

## 🧠 How It Works

Our app uses a combination of traditional ML models and ensemble techniques. Here's a sneak peek at our ensemble prediction method:

```python
def new_ensemble_prediction(model_results):
    confidence_scores = calculate_confidence(model_results)
    weights = model_results['Backtesting_Accuracy'] * confidence_scores
    weighted_predictions = (model_results['Prediction'] * weights).sum()
    total_weight = weights.sum()
    final_prediction = weighted_predictions / total_weight
    return 1 if final_prediction >= 0.5 else 0
```

📊 Sample Output
CopyPrediction for Nifty 50 on 2024-08-15:
Random Forest: Backtest Accuracy: 52.34%, Prediction: 📈 Bullish
KNN: Backtest Accuracy: 48.67%, Prediction: 📉 Bearish
Decision Tree: Backtest Accuracy: 51.12%, Prediction: 📈 Bullish
XGBoost: Backtest Accuracy: 53.89%, Prediction: 📈 Bullish

Final Prediction: 📈 Bullish
⚠️ Disclaimer
This tool is for educational and informational purposes only. Always conduct your own research and consult with a qualified financial advisor before making investment decisions.
🤝 Contributing
We welcome contributions! If you have ideas for improvements or bug fixes, please open an issue or submit a pull request.
📬 Contact
For more projects and information, visit my portfolio: [https://kartikey-vyas-ds.github.io/]

Created with ❤️ by Kartikey Vyas

## Contact

If you have any feedback/are interested in collaborating, please reach out to me at [<img height="40" src="https://img.icons8.com/color/48/000000/linkedin.png" height="40em" align="center" alt="Follow Kartikey on LinkedIn" title="Follow Kartikey on LinkedIn"/>](https://www.linkedin.com/in/kartikey-vyas-2a29b9273) &nbsp; <a href="mailto:kvsvyas@gmail.com"> <img height="40" src="https://img.icons8.com/fluent/48/000000/gmail.png" align="center" />





## License

[MIT](https://choosealicense.com/licenses/mit/)