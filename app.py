from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import requests
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import numpy as np

API_KEY = 'V7G26Y714H646OQF'

app = Flask(__name__)
app.secret_key = 'Abdullah@1'


def fetch_stock_data(symbol):
    endpoint = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}'
    response = requests.get(endpoint)

    if response.status_code == 200:
        data = response.json()
        time_series_data = data['Time Series (Daily)']
        df = pd.DataFrame.from_dict(time_series_data, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.astype(float)
        return df
    else:
        print('Error fetching data. Please check your API key and symbol.')
        return None





def calculate_technical_indicators(df):
    df['5-day MA'] = df['4. close'].rolling(window=5).mean()

    delta = df['4. close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    df['RSI'] = rsi

    return df


def create_target_variable(df):
    # Create the target variable (Next Day's Closing Price) by shifting the '4. close' column by one day
    df['Next Day\'s Closing Price'] = df['4. close'].shift(-1)
    return df


@app.route('/', methods=['GET', 'POST'])
def index():
    stock_data = None
    selected_company = None  # Initialize selected_company as None
    if request.method == 'POST':
        name = request.form['name']
        password = request.form['password']

        # Replace 'your_username' and 'your_password' with the actual username and password to login.
        if name == 'abdullah' and password == 'Abdullah@1':
            session['username'] = name
            selected_company = request.form['stockCompany']  # Get the selected stock company from the form
        else:
            return render_template('login.html', error='Invalid credentials')




    if 'username' in session:
        if selected_company is not None:
            stock_data = fetch_stock_data(selected_company)  
            if stock_data is not None:
                print(f'Historical Stock Data for {selected_company}:')  # Use selected_company instead of stock_symbol
                print(stock_data)

            # Handle Missing Values - Filling with Mean
            stock_data = stock_data.fillna(stock_data.mean())

            # Feature Engineering - Technical Indicators
            stock_data = calculate_technical_indicators(stock_data)

            # Data Exploration
            print('\nData Exploration:')
            print(stock_data.describe())

            # Check Data Scaling
            print('\nData Scaling:')
            column_ranges = stock_data.max() - stock_data.min()
            print(column_ranges)

            # Normalize the Data using Min-Max Scaling
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(stock_data)

            # Create a DataFrame from the scaled data
            scaled_df = pd.DataFrame(scaled_data, columns=stock_data.columns, index=stock_data.index)
            # Verify that the data does not lose its original distribution
            print('\nOriginal Data:')
            print(stock_data.head())

            print('\nNormalized Data:')
            print(scaled_df.head())

            stock_data = create_target_variable(stock_data)

            # Data Preprocessing - Train-Test Split
            features = stock_data.drop(columns=['Next Day\'s Closing Price'])
            target = stock_data['Next Day\'s Closing Price']

            # Drop rows with NaN values in the target variable
            stock_data = stock_data.dropna(subset=['Next Day\'s Closing Price'])

            # Use an imputer in a pipeline to handle missing values for features
            feature_pipeline = make_pipeline(SimpleImputer(strategy='median'), MinMaxScaler())
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

            # Apply imputation on features
            X_train = feature_pipeline.fit_transform(X_train)
            X_test = feature_pipeline.transform(X_test)

            
            X_train = X_train[~np.isnan(y_train)]
            y_train = y_train.dropna()

            
            model = RandomForestRegressor(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            feature_importance = model.feature_importances_
            sorted_indices = np.argsort(feature_importance)[::-1]

           
            n_top_features = 5
            selected_features = features.columns[sorted_indices][:n_top_features]
            selected_importance = feature_importance[sorted_indices][:n_top_features]

           
            print("Selected Features and Importance Scores:")
            for feature, importance in zip(selected_features, selected_importance):
                print(f"{feature}: {importance:.4f}")

            mse = mean_squared_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mae = mean_absolute_error(y_test, y_pred)
            threshold = 1.5 # Set your threshold here
            decision = "Buy" if mae < threshold else "Not Buy"
            evaluation_results = {
              'mse': mse,
              'rmse': rmse,
              'mae': mae,
              'decision': decision
           }

            train_data = pd.concat([pd.DataFrame(X_train, columns=features.columns), pd.Series(y_train, name='Next Day\'s Closing Price')], axis=1)
            test_data = pd.concat([pd.DataFrame(X_test, columns=features.columns), pd.Series(y_test, name='Next Day\'s Closing Price')], axis=1)
            train_data.to_csv('train_dataset.csv', index=False)
            test_data.to_csv('test_dataset.csv', index=False)
            num_test_records = len(test_data)
            num_train_records = len(train_data)
            comparison_data = pd.DataFrame({'Date': y_test.index, 'Actual Value': y_test, 'Predicted Value': y_pred})
            comparison_data.to_csv('comparison.csv', index=False)
            print(f'Number of records in test_data: {num_test_records}')
            print(f'Number of records in train_data: {num_train_records}')
            plt.figure(figsize=(10, 6))
            plt.plot(y_test.index, y_test, label='Actual Closing Price')
            plt.plot(y_test.index, y_pred, label='Predicted Closing Price')
            plt.xlabel('Date')
            plt.ylabel('Closing Price')
            plt.title(f'Actual vs. Predicted Closing Prices for {selected_company}')
            plt.legend()
            plt.grid(True)
            plot_path ='static/prediction_plot.png'
            plt.savefig(plot_path)
            plt.close()

            return render_template('stock_data.html', plot_url='static/prediction_plot.png', evaluation_results=evaluation_results, selected_company=selected_company, comparison_data=comparison_data)
        else:
        
            # If there was an issue fetching the data, show an error message or redirect to login page.
            return render_template('login.html', error='Error fetching data. Please try again later.')
    else:
        # If the user is not logged in, show the login page.
        return render_template('login.html')
@app.route('/stock_data/<string:company>', methods=['GET'])
def stock_data(company):
    # Check if the user is logged in
    if 'admin' in session:
        stock_data = fetch_stock_data(company)
        if stock_data is not None:
           

            return render_template('stock_data.html', plot_url='static/prediction_plot.png', evaluation_results=evaluation_results, selected_company=company)
        else:
            return render_template('login.html', error='Error fetching data. Please try again later.')
    else:
        return render_template('login.html')


@app.route('/stock_plot')
def show_stock_plot():
    return render_template('stock_plot.html', plot_url='static/stock_plot.png')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))

if __name__ == "__main__":
        app.run(debug=True)
