import gradio as gr
import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import yfinance as yf
from datetime import date, timedelta


def get_indian_stock_tickers():
    indian_tickers = {"Bank Nifty":"^NSEBANK", "NIFTY MIDCAP 50" : "^NSEMDCP50","NIFTY IT": "^CNXIT","Nifty 50": "^NSEI","Nifty Pharma":"^CNXPHARMA"}
    return indian_tickers

def get_historical_data(ticker, start_date, end_date):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        return stock_data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def stock_data_preprocessing(data):
    data['Date'] = pd.to_datetime(data.index)
    data.reset_index(drop=True, inplace=True)
    del data['Adj Close']
    data = data.sort_values('Date')
    data['Tomorrow'] = data['Close'].shift(-1)
    data['Target'] = (data['Tomorrow'] > data['Close']).astype(int)
    return data

def stock_data_feature_engineering(data):
    horizons = [2, 5, 30, 60, 1000]
    for horizon in horizons:
        rolling_avgs = data['Close'].rolling(horizon).mean()
        ratio_column = f"Close_Ratio_{horizon}"
        data[ratio_column] = data['Close'] / rolling_avgs
        trend_column = f"Trend_{horizon}"
        data[trend_column] = data["Target"].shift(1).rolling(horizon).sum()
    data['RSI'] = ta.rsi(data.Close, length=15)
    data['EMAF'] = ta.ema(data.Close, length=20)
    data['EMAM'] = ta.ema(data.Close, length=100)
    data['EMAS'] = ta.ema(data.Close, length=150)
    data[['MACD', 'Signal', 'Histogram']] = ta.macd(data.Close)
    data = data.dropna()
    return data

def feature_selection(train_data):
    X = train_data[[col for col in train_data.columns if col not in ["Tomorrow","Target","Date"]]]
    y = train_data["Target"]
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    feature_importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    top_k_features = feature_importance_df.head(10)['Feature'].tolist()
    return top_k_features

def predict(train, test, predictors, model):
    model.fit(train[predictors], train['Target'])
    preds = model.predict_proba(test[predictors])[:, 1]
    preds[preds >= 0.6] = 1 
    preds[preds < 0.6] = 0  
    preds = pd.Series(preds, index=test.index, name="Predictions")
    return preds

def backtest(data, model, predictors, start=1250, step=250):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:i+step].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

def train_randomforest(train_data, test_data, predictors):
    rf_model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=19)
    predictions = backtest(train_data, rf_model, predictors)
    backtest_acc = accuracy_score(train_data.loc[predictions.index, "Target"], predictions)
    final_predictions = predict(train_data, test_data, predictors, rf_model)
    row = {
        'Model_Name': 'Random Forest',
        'Backtesting_Accuracy': backtest_acc,
        'Prediction': final_predictions.iloc[0] if isinstance(final_predictions, pd.Series) else final_predictions
    }
    pred_df = pd.DataFrame([row])  # Ensure it's a DataFrame with one row
    return pred_df

def train_knn(train_data, test_data, predictors):
    knn_model = KNeighborsClassifier(algorithm='auto', n_neighbors=3, weights='distance')
    
    # Use the last 20% of train_data for backtesting
    split_index = int(len(train_data) * 0.8)
    train_subset = train_data.iloc[:split_index]
    backtest_subset = train_data.iloc[split_index:]
    
    knn_model.fit(train_subset[predictors], train_subset["Target"])
    backtest_predictions = knn_model.predict(backtest_subset[predictors])
    backtest_acc = accuracy_score(backtest_subset["Target"], backtest_predictions)
    
    # Retrain on full training data for final prediction
    knn_model.fit(train_data[predictors], train_data["Target"])
    final_predictions = knn_model.predict(test_data[predictors])
    
    pred_df = pd.DataFrame({
        'Model_Name': ['KNN'],
        'Backtesting_Accuracy': [backtest_acc],
        'Prediction': [final_predictions[0] if isinstance(final_predictions, np.ndarray) else final_predictions]
    })
    return pred_df

def train_decision_tree(train_data, test_data, predictors):
    dt_model = DecisionTreeClassifier(max_depth=9, min_samples_leaf=1, min_samples_split=2, random_state=1)
    predictions = backtest(train_data, dt_model, predictors)
    backtest_acc = accuracy_score(train_data.loc[predictions.index, "Target"], predictions)
    final_predictions = predict(train_data, test_data, predictors, dt_model)
    row = {
        'Model_Name': 'Decision Tree',
        'Backtesting_Accuracy': backtest_acc,
        'Prediction': final_predictions.iloc[0] if isinstance(final_predictions, pd.Series) else final_predictions
    }
    pred_df = pd.DataFrame([row])  # Ensure it's a DataFrame with one row
    return pred_df

def train_xgboost(train_data, test_data, predictors):
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(train_data[predictors], train_data['Target'])
    predictions = xgb_model.predict(train_data[predictors])
    backtest_acc = accuracy_score(train_data['Target'], predictions)
    final_predictions = xgb_model.predict(test_data[predictors])
    row = {
        'Model_Name': 'XGBoost',
        'Backtesting_Accuracy': backtest_acc,
        'Prediction': final_predictions[0]
    }
    pred_df = pd.DataFrame([row])
    return pred_df
def calculate_confidence(model_results):
    # Calculate confidence based on the difference between prediction probability and 0.5
    confidence_scores = []
    for _, row in model_results.iterrows():
        if row['Model_Name'] == 'Random Forest':
            confidence = abs(row['Prediction'] - 0.5) * 2  # Scale to [0, 1]
        elif row['Model_Name'] in ['KNN', 'Decision Tree', 'XGBoost']:
            confidence = 1 if row['Prediction'] in [0, 1] else 0.5  # Binary predictions
        confidence_scores.append(confidence)
    return confidence_scores

def new_ensemble_prediction(model_results):
    confidence_scores = calculate_confidence(model_results)
    weights = model_results['Backtesting_Accuracy'] * confidence_scores
    weighted_predictions = (model_results['Prediction'] * weights).sum()
    total_weight = weights.sum()
    final_prediction = weighted_predictions / total_weight
    return 1 if final_prediction >= 0.5 else 0

def ensemble_prediction(results):
    results.loc[results["Prediction"] == 0, "Prediction"] = 2
    final_prediction_numerator = (results["Backtesting_Accuracy"] * results["Prediction"]).sum()
    final_prediction_denominator = results["Backtesting_Accuracy"].sum()
    final_prediction = final_prediction_numerator / final_prediction_denominator
    if final_prediction >= 1.2:  # This is equivalent to 0.6 for the binary case
        return 1
    else:
        return 0

def predict_market(ticker, prediction_date_str):
    try:
        prediction_date = pd.to_datetime(prediction_date_str).date()
    except ValueError:
        return "Invalid date format. Please use YYYY-MM-DD.", None

    today = date.today()
    
    if prediction_date <= today:
        return "Prediction date must be in the future.", None
    
    if prediction_date > today + timedelta(days=7):
        return "Prediction can only be made up to 7 days in the future.", None
    
    if prediction_date.weekday() >= 5:
        return "Prediction date must be a weekday (assumed trading day).", None

    try:
        # Calculate the last available trading day (assuming it's the previous weekday)
        last_trading_day = today
        while last_trading_day.weekday() >= 5:
            last_trading_day -= timedelta(days=1)

        # Use the last trading day as the end of our training data
        train_end_date = last_trading_day
        
        indian_stock_tickers = get_indian_stock_tickers()
        train_start_date = '2000-01-01'

        og_ticker_data = get_historical_data(indian_stock_tickers.get(ticker),
                                             train_start_date,
                                             train_end_date)

        if og_ticker_data.empty:
            return f"No historical data available for {ticker}.", None

        ticker_data = og_ticker_data.copy()
        ticker_data = stock_data_preprocessing(ticker_data)
        ticker_data = stock_data_feature_engineering(ticker_data)

        # Use all available data for training
        train_data = ticker_data
        
        # Create a dummy row for the prediction date
        last_row = train_data.iloc[-1].copy()
        last_row.name = prediction_date
        test_data = pd.DataFrame([last_row])
        
        feature_selected = feature_selection(train_data)

        sc = MinMaxScaler(feature_range=(0, 1))
        temp_col = feature_selected + ['Target']
        ticker_data = pd.DataFrame(sc.fit_transform(ticker_data[temp_col]), index=ticker_data["Date"])
        ticker_data.columns = temp_col

        train_data = ticker_data
        test_data = pd.DataFrame(sc.transform(test_data[temp_col]), index=[prediction_date], columns=temp_col)

        model_results = pd.DataFrame(columns=['Model_Name', 'Backtesting_Accuracy', 'Prediction'])

        rf_result = train_randomforest(train_data, test_data, feature_selected)
        model_results = pd.concat([model_results, rf_result], ignore_index=True, sort=False)

        KNN_result = train_knn(train_data, test_data, feature_selected)
        model_results = pd.concat([model_results, KNN_result], ignore_index=True, sort=False)

        Decisiontree_result = train_decision_tree(train_data, test_data, feature_selected)
        model_results = pd.concat([model_results, Decisiontree_result], ignore_index=True, sort=False)

        XGBoost_result = train_xgboost(train_data, test_data, feature_selected)
        model_results = pd.concat([model_results, XGBoost_result], ignore_index=True, sort=False)

        final_prediction = new_ensemble_prediction(model_results)

        result = f"Prediction for {ticker} on {prediction_date}:\n"
        for _, row in model_results.iterrows():
            emoji_text = "📈 Bullish" if row['Prediction'] == 1 else "📉 Bearish"
            result += f"{row['Model_Name']}: Backtest Accuracy: {row['Backtesting_Accuracy']:.2%}, Prediction: {emoji_text}\n"
        
        final_emoji_text = "📈 Bullish" if final_prediction == 1 else "📉 Bearish"
        result += f"\nFinal Prediction: {final_emoji_text}"

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x=og_ticker_data.index, y="Close", data=og_ticker_data)
        plt.xlabel("Date")
        plt.ylabel("Closing Price")
        plt.title(f"Historical Trend for {ticker}")

        return result, fig

    except Exception as e:
        error_message = f"An error occurred: {str(e)}\n"
        error_message += "Please try again or contact support if the problem persists."
        return error_message, None

iface = gr.Interface(
    fn=predict_market,
    inputs=[
        gr.Dropdown(choices=list(get_indian_stock_tickers().keys()), label="Select Indian Stock Ticker"),
        gr.Textbox(label="Select Prediction Date (YYYY-MM-DD)", value=(date.today() + timedelta(days=1)).strftime("%Y-%m-%d"))
    ],
    outputs=[
        gr.Textbox(label="Prediction Result"),
        gr.Plot(label="Historical Trend")
    ],
    title="Market Momentum: Advanced ML-based Prediction",
    description="""
    Predict market momentum using an ensemble of advanced machine learning models.
    
    Instructions:
    1. Select an Indian stock ticker from the dropdown.
    2. Enter a prediction date (YYYY-MM-DD format) 1 to 7 days in the future.
    
    Features:
    - Utilizes four different ML models: Random Forest, KNN, Decision Tree, and XGBoost.
    - Implements a sophisticated ensemble method considering both model accuracy and prediction confidence.
    - Provides individual predictions from each model along with their backtest accuracies.
    - Offers a final ensemble prediction for more robust results.
    
    Notes:
    - The prediction date must be a weekday (assumed trading day).
    - Predictions can only be made for 1 to 7 days in the future.
    - 📈 Bullish indicates a positive momentum prediction.
    - 📉 Bearish indicates a negative momentum prediction.
    
    The app uses historical data up to the most recent trading day to predict market momentum for the selected future date.
    
    Disclaimer:
    This tool is for educational and informational purposes only. The predictions provided are based on historical data and should not be considered as financial advice. Stock market investments carry risks, and past performance does not guarantee future results. Always conduct your own research and consult with a qualified financial advisor before making investment decisions. The creators and operators of this tool are not responsible for any financial losses incurred based on these predictions.
    
    For more projects and information, visit my portfolio: https://kartikey-vyas-ds.github.io/
    """
)

iface.launch()