
# Importing required libraries
import streamlit as st
from datetime import date, timedelta
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import numpy as np
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override() # <== that's all it takes :-)


# Set page title
st.set_page_config(page_title="StockPulse: Mapping Market Momentum",
                   page_icon="🧊",
                   layout="wide")

# Function to display calendar-like date selection
def calendar_input():
  selected_date = st.date_input("Select Date", value=None)
  # Additional logic for styling or custom functionalities (optional)
  return selected_date

# Define function to get list of Indian stock tickers
def get_indian_stock_tickers():
    # You may need to fetch this list from a reliable source or API
    # For demonstration purpose, I'll provide some example tickers
    indian_tickers = {"Reliance":"RELIANCE.NS", "TCS" : "TCS.NS","Infosys": "INFY.NS","Nifty 50": "^NSEI"}

    return indian_tickers

# Define function to fetch historical stock data
def get_historical_data(ticker, start_date, end_date):
    try:
        stock_data = pdr.get_data_yahoo(ticker, start=start_date, end=end_date, progress=False)
        return stock_data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error

# Define function to plot historical trend line
def plot_trend_line(data,selected_ticker):
  # Check if data is downloaded successfully
  if data.empty:
    st.error("Error downloading data for " + selected_ticker)
  else:
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # Prepare data for visualization
    data['Date']=data.index
    data.reset_index(drop=True,inplace=True)

    # Historical Trend Line with Seaborn
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size for better visualization
    sns.lineplot(x="Date", y="Close", data=data)  # Use seaborn lineplot
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.title(f"Historical Trend for {selected_ticker}")
    st.pyplot(fig)

def stock_data_preprocessing(data):

  # 1. Create Date Column
  data['Date']=pd.to_datetime(data.index)
  data.reset_index(drop=True,inplace=True)

  # 2. Deleting Adj Close column
  del data['Adj Close']

  # 3. Sorting data
  data=data.sort_values('Date')

  # 4. Creating Target Column
  data['Tomorrow']=data['Close'].shift(-1)
  data['Target']=(data['Tomorrow'] > data['Close']).astype(int)

  return data

def stock_data_feature_engineering(data):

  # 1. Create Trend and Ratio variables
  horizons=[2,5,30,60,1000]

  for horizon in horizons:
    rolling_avgs=data['Close'].rolling(horizon).mean()
    ratio_column=f"Close_Ratio_{horizon}"
    data[ratio_column]=data['Close']/rolling_avgs

    trend_column=f"Trend_{horizon}"
    data[trend_column]=data["Target"].shift(1).rolling(horizon).sum()

  # 2. Create technical indicators - RSI
  data['RSI']=ta.rsi(data.Close, length=15)
  data['EMAF']=ta.ema(data.Close, length=20)
  data['EMAM']=ta.ema(data.Close, length=100)
  data['EMAS']=ta.ema(data.Close, length=150)
  data[['MACD','Signal','Histogram']]=ta.macd(data.Close)

  # # 3. Create lags
  # lag=30
  # for i in range(1,lag+1):
  #   lag_col=f'Close_lag_{i}'
  #   data[lag_col]=data["Close"].shift(i)

  # 4. Remove NA rows
  data=data.dropna()

  return data

def feature_selection(train_data):
  # Replace 'X' and 'y' with your actual feature matrix and target vector
  X = train_data[[col for col in train_data.columns if col not in ["Tomorrow","Target","Date"]]]  # Feature matrix
  y = train_data["Target"]      # Target vector

  # Train Random Forest model
  rf = RandomForestClassifier(n_estimators=100, random_state=42)
  rf.fit(X, y)

  # Retrieve feature importances
  feature_importances = rf.feature_importances_

  # Create a DataFrame with feature names and their importances
  feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

  # Sort features by importance (descending order)
  feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

  # Select top k features (e.g., top 10 features)
  top_k_features = feature_importance_df.head(10)['Feature'].tolist()

  # Print selected features
  print("Top 10 Features:")
  print(top_k_features)

  return top_k_features

def predict(train,test,predictors,model):
    model.fit(train[predictors],train['Target'])
    preds=model.predict_proba(test[predictors])[:,1]
    preds[preds>=0.6]=1
    preds[preds<0.6]=0
    preds=pd.Series(preds,index=test.index,name="Predictions")
    combined=pd.concat([test['Target'],preds],axis=1)
    return combined

def backtest(data,model,predictors, start=1250, step=250):
    all_predictions=[]

    for i in range(start,data.shape[0],step):
      train=data.iloc[0:i].copy()
      test=data.iloc[i:i+step].copy()
      predictions=predict(train,test,predictors,model)
      all_predictions.append(predictions)

    return pd.concat(all_predictions)

def train_randomforest(train_data,test_data,predictors):

  # Training Model
  rf_model=RandomForestClassifier(n_estimators=200,min_samples_split=50,random_state=19)

  # Backtesting
  predictions=backtest(train_data,rf_model,predictors)
  backtest_acc=accuracy_score(predictions["Target"],predictions["Predictions"])

  # Prediction
  final_predictions=predict(train_data.iloc[-1250:],test_data,predictors,rf_model)

  # Row
  row={
      'Model_Name':'Random Forest',
      'Backtesting_Accuracy':backtest_acc,
      'Prediction':final_predictions['Predictions'],
      'Actual':final_predictions['Target']
  }
  pred_df=pd.DataFrame(row)

  return pred_df

def train_xgboost(train_data,test_data,predictors):

  # Training Model
  xgb_model=xgb_classifier = XGBClassifier(objective='binary:logistic',  # Binary classification
    use_label_encoder=False,  # Prevents warnings about deprecated label encoder
    eval_metric='logloss')

  # Backtesting
  predictions=backtest(train_data,xgb_model,predictors)
  backtest_acc=accuracy_score(predictions["Target"],predictions["Predictions"])

  # Prediction
  final_predictions=predict(train_data.iloc[-1250:],test_data,predictors,xgb_model)

  # Row
  row={
      'Model_Name':'XG Boost',
      'Backtesting_Accuracy':backtest_acc,
      'Prediction':final_predictions['Predictions'],
      'Actual':final_predictions['Target']
  }
  pred_df=pd.DataFrame(row)

  return pred_df

def create_sequence(dataset):
  sequences = []
  labels = []

  start_idx = 0
  include_cols_indices = [col_idx for col_idx, col_name in enumerate(dataset.columns) if col_name not in "Target"]

  for stop_idx in range(30,len(dataset)): # Selecting 30 rows at a time
    sequences.append(dataset.iloc[start_idx:stop_idx,include_cols_indices])
    labels.append(dataset.iloc[stop_idx]["Target"])
    start_idx += 1
  return (np.array(sequences),np.array(labels))

def predict_lstm(train, test):
    # X_train, y_train = create_sequence(train)
    # X_test, y_test = create_sequence(test)
    y_train = train["Target"]
    X_train = train.drop(columns=["Target"])

    y_test = test["Target"]
    X_test = test.drop(columns=["Target"])

    # Reshape Input Data for LSTM
    X_train_reshaped = np.array(X_train).reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test_reshaped = np.array(X_test).reshape(X_test.shape[0], 1, X_test.shape[1])

    try:
        # Define LSTM model architecture
        model = Sequential()
        model.add(LSTM(units=50, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
        model.add(Dense(units=1, activation='sigmoid'))  # Output layer with sigmoid activation for binary classification

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_split=0.2, shuffle=True, verbose=0)

        # Evaluate the model on test data
        preds = model.predict(X_test_reshaped)
        preds = (preds > 0.5).astype(int)  # Convert probabilities to binary predictions

        preds = pd.Series(preds.flatten(), index=test.index, name="Predictions")
        combined = pd.concat([test['Target'], preds], axis=1)

        return combined
    except Exception as e:
        st.error(f"Error during LSTM prediction: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error
    
def backtest_lstm(data, start=1250, step=250):
    all_predictions=[]

    for i in range(start,data.shape[0],step):
      train=data.iloc[0:i].copy()
      test=data.iloc[i:i+step].copy()
      predictions=predict_lstm(train,test)
      # predictions = predictions.fillna(0)
      all_predictions.append(predictions)

    return pd.concat(all_predictions)

def train_lstm(train_data,test_data):
  # Backtesting
  predictions=backtest_lstm(train_data)
  backtest_acc=accuracy_score(predictions["Target"],predictions["Predictions"])

  # Prediction
  final_predictions=predict_lstm(train_data.iloc[-1250:],test_data)

  # Row
  row={
      'Model_Name':'LSTM',
      'Backtesting_Accuracy':backtest_acc,
      'Prediction':final_predictions['Predictions'],
      'Actual':final_predictions['Target']
  }
  pred_df=pd.DataFrame(row)

  return pred_df

def ensemble_prediction(results):
  # converting 0 to 2
  results.loc[results["Prediction"]==0,"Prediction"]=2

  # Calculating final prediction
  final_prediction_numerator=(results["Backtesting_Accuracy"] * results["Prediction"]).sum()
  final_prediction_denominator=results["Backtesting_Accuracy"].sum()
  final_prediction=round(final_prediction_numerator/final_prediction_denominator,0)

  # converting 2 to 0
  if final_prediction==2:
    final_prediction=0

  return final_prediction

def display_conditional_image(value):
  if value == 0:
    st.image("https://drive.google.com/file/d/1FLFVqZG6J6iMvPgFtA4eEXwwlofctGOL/view?usp=sharing", caption="Bearish")
  else:
    st.image("https://drive.google.com/file/d/1OxBNubZUNj_ItRcfzoEB9nM4NUbhe7SW/view?usp=sharing", caption="Bullish")

def display_model_details(model_df, col):
  with col:
    st.header(model_df["Model_Name"].iloc[0])
    st.write(f"Backtest Accuracy: {model_df['Backtesting_Accuracy'].iloc[0]:.2%}")
    display_conditional_image(model_df['Prediction'].iloc[0])
    # Access and display other relevant columns from model_df here
    
# Function to display progress bar
def show_progress_bar(progress):
    st.write(f"Training Progress: {progress}%")
    st.progress(progress)


def main():

    st.title("StockPulse: Mapping Market Momentum")

    # Get list of Indian stock tickers
    indian_stock_tickers = get_indian_stock_tickers()

    # Dropdown to select stock ticker
    selected_ticker = st.selectbox("Select Indian Stock Ticker", list(indian_stock_tickers.keys()))

    # Default date set to yesterday (one day before today)
    today = date.today()
    default_date = today - timedelta(days = 1)

    # Display the calendar-like date selection
    prediction_date=st.date_input(label="Select Prediction Date",value=default_date)

    # Button to fetch data and plot trend line
    if st.button("Go"):
      # Input Parameter
      train_start_date='2000-01-01'
      train_end_date=prediction_date- timedelta(days = 1)
      test_start_date=prediction_date
      test_end_date=prediction_date

      st.write(f"Fetching historical data for **{selected_ticker}**...")
      og_ticker_data = get_historical_data(indian_stock_tickers.get(selected_ticker),
                                       train_start_date,
                                       pd.to_datetime(test_end_date)+pd.DateOffset(days=1))

      ticker_data=og_ticker_data

      # Preprocessing
      st.write("Data preprocessing...")
      ticker_data=stock_data_preprocessing(ticker_data)

      # Feature Engineering
      st.write("Feature Enginnering...")
      ticker_data=stock_data_feature_engineering(ticker_data)

      # Feature Engineering
      test_start_date=ticker_data["Date"].max()
      test_end_date=ticker_data["Date"].max()
      train_data=ticker_data[ticker_data["Date"]<test_start_date]
      st.write(f"Predicting for **{test_end_date}**...")

      # Feature Selection
      st.write("Feature Selection...")
      feature_selected=feature_selection(train_data)

      # Feature Selection
      st.write(f"Selected Features : **{ ', '.join(map(str, feature_selected))}**")

      # Scaling
      st.write(f"Scaling selected features...")
      sc = MinMaxScaler(feature_range=(0,1))

      temp_col=feature_selected+['Target']
      ticker_data=pd.DataFrame(sc.fit_transform(ticker_data[temp_col]),index=ticker_data["Date"])
      ticker_data.columns=temp_col

      # Train Test Split
      train_data=ticker_data[(ticker_data.index < test_end_date)]
      test_data=ticker_data[(ticker_data.index == test_end_date)]

      model_results = pd.DataFrame(columns=['Model_Name', 'Backtesting_Accuracy', 'Prediction','Actual'])

      # Random Forest
      st.write(f"Model Training: Random Forest...")
      rf_result=train_randomforest(train_data,test_data,feature_selected)
      model_results = pd.concat([model_results, rf_result], ignore_index=True)

      # XG Boost
      st.write(f"Model Training: XG boost...")
      show_progress_bar(50)  # Update progress bar
      xgb_result = train_xgboost(train_data, test_data, feature_selected)
      model_results = pd.concat([model_results, xgb_result], ignore_index=True) # Select specific columns

      # LSTM
      st.write(f"Model Training: LSTM...")
      lstm_result=train_lstm(train_data,test_data)
      model_results = pd.concat([model_results, lstm_result], ignore_index=True)  # Select specific columns

      # Display title or additional information at the top (optional)
      st.title("Model Predictions")

      # Layout with three columns
      col1, col2, col3 = st.columns(3)

      for i, row in model_results.iterrows():
        model_df = model_results.iloc[[i]]  # Create DataFrame for a single model (row)
        display_model_details(model_df.copy(), col1 if i % 3 == 0 else col2 if i % 3 == 1 else col3)  # Optional function call

      # Layout with three columns
      col4, col5, col6 = st.columns(3)

      with col5:
        st.header("Final Prediction")
        display_conditional_image(ensemble_prediction(model_results))


      # Display title or additional information at the top (optional)
      st.title("Statistics")
      # ... other app elements

      st.write("Plotting historical trend line...")
      plot_trend_line(og_ticker_data,selected_ticker)

if __name__ == "__main__":
    main()
