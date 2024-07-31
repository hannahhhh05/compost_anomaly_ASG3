import pandas as pd
import os
import glob
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import statsmodels
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import os
os.getcwd()

def create_gspread_client(credentials_path):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_file(credentials_path, scopes=scope)
    # Authorize the gspread client
    client = gspread.authorize(creds)
    return client

def authenticate_drive(credentials_path):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    # Load the credentials
    creds = Credentials.from_service_account_file(credentials_path, scopes=scope)
    # Build the Google Drive service
    drive_service = build('drive', 'v3', credentials=creds)
    return drive_service

def get_sheets_in_folder(drive_service, folder_id):
    query = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.spreadsheet'"
    results = drive_service.files().list(q=query, pageSize=1000, fields="files(id, name)").execute()
    items = results.get('files', [])
    print(items)
    return items

def read_and_concat_sheets(client, file_id, header_row=1):
    spreadsheet = client.open_by_key(file_id)
    all_sheets_data = []
    for sheet in spreadsheet.worksheets():
        sheet_data = pd.DataFrame(sheet.get_all_values())
        # Check if the first row has empty values
        if any(pd.isna(sheet_data.iloc[header_row-1])):
            # If the header row has empty values, use the next row as the header
            sheet_data.columns = sheet_data.iloc[header_row]
        else:
            # If the header row is not empty, use the header row as the header
            sheet_data.columns = sheet_data.iloc[header_row-1]
        sheet_data.reset_index(drop=True, inplace=True)  # Reset index
        # Define the schema of the final merged table
        final_columns = ['Timestamp', 'Number of Worms (non-counted)', 'Phosphorous01', 'Phosphorous02',
                         'Nitrogen01', 'Nitrogen02', 'Potassium01', 'Potassium02', 'Light Intensity',
                         'Temp01', 'Hum01', 'Heat01', 'SoilM01', 'SoilM02', 'Buzzer', 'pH Rod 1', 'pH Rod 2']
        # Drop any extra columns from the individual table
                # Drop any duplicate columns
        sheet_data = sheet_data.loc[:, ~sheet_data.columns.duplicated()]
        sheet_data = sheet_data.reindex(columns=final_columns, fill_value=None)
        print(sheet_data.shape)
        sheet_name = sheet.title
        print("Sheet Name:", sheet_name)
        all_sheets_data.append(sheet_data)
    # Concatenate all sheet data into a single DataFrame
    concatenated_data = pd.concat(all_sheets_data, ignore_index=True)
    return concatenated_data

def load_and_concat_all_sheets_in_centers(base_directory_id, credentials_path):
    try:
        client = create_gspread_client(credentials_path)
        drive_service = authenticate_drive(credentials_path)
        
        compost = []
        
        center_folders = drive_service.files().list(q=f"'{base_directory_id}' in parents and mimeType='application/vnd.google-apps.folder'",
                                                    fields="files(id, name)").execute().get('files', [])
        
        for center_folder in center_folders:
            sheet_files = get_sheets_in_folder(drive_service, center_folder['id'])
            for sheet_file in sheet_files:
                sheet_data = read_and_concat_sheets(client, sheet_file['id'], header_row=1)
                sheet_data['Location'] = center_folder['name'].split('_')[1]
                compost.append(sheet_data)
        
        return pd.concat(compost, ignore_index=True)
    except HttpError as error:
        print(f"An error occurred: {error}")
        return None

# Example usage
base_directory_id = '1uH8e33HJQG4v8BmCZ_EJYmGH-VmWI547'  # The ID of the main folder containing center folders
credentials_path = r'C:\Users\ADMIN\OneDrive\Desktop\DSCP\npk_forecasting_final\streamlit-npkcomposter-anomaly-71e49f3f60a9.json' 
compost = load_and_concat_all_sheets_in_centers(base_directory_id, credentials_path)
print(compost.shape)

pd.set_option('display.max_column', None)
compost.head()

compost.info()

filtered_data = compost.copy()

for column in filtered_data.columns:
    # Calculate the percentage of NaN values in the column
    nan_percentage = filtered_data[column].isna().mean()
    print(nan_percentage)

# Calculate the threshold for dropping columns
threshold = len(filtered_data) * 0.5

# Iterate over each column
for column in filtered_data.columns:
    # Calculate the percentage of NaN values in the column
    nan_percentage = filtered_data[column].isna().mean()
    
    # If NaN percentage >= 50%, drop the column
    if nan_percentage >= 0.5:
        filtered_data.drop(column, axis=1, inplace=True)

# Print the updated DataFrame shape
print("Shape after dropping columns with NaN or null values >= 50%:", filtered_data.shape)


filtered_data.head()

filtered_data.info()

filtered_data = filtered_data[~filtered_data['Timestamp'].astype(str).str.contains('Unit|Timestamp', case=False)]
filtered_data = filtered_data[filtered_data['Timestamp'].notna()]

columns_to_convert = ['Phosphorous01', 'Phosphorous02', 'Nitrogen01', 'Nitrogen02', 
                        'Potassium01', 'Potassium02', 'Temp01', 'Hum01', 'Heat01', 
                        'SoilM01', 'SoilM02', 'Number of Worms (non-counted)']

# Attempt to convert to float, replacing non-numeric values with NaN
filtered_data[columns_to_convert] = filtered_data[columns_to_convert].apply(pd.to_numeric, errors="coerce")

# Check the data types
print(filtered_data.dtypes)
print(filtered_data.info())

# Calculate the threshold for dropping columns
threshold = len(filtered_data) * 0.5

# Iterate over each column
for column in filtered_data.columns:
    # Calculate the percentage of NaN values in the column
    nan_percentage = filtered_data[column].isna().mean()
    
    # If NaN percentage >= 50%, drop the column
    if nan_percentage >= 0.5:
        filtered_data.drop(column, axis=1, inplace=True)

# Print the updated DataFrame shape
print("Shape after dropping columns with NaN or null values >= 50%:", filtered_data.shape)

# Feature Engineering

filtered_data['Timestamp'] = pd.to_datetime(filtered_data['Timestamp'])
filtered_data['Hour'] = filtered_data['Timestamp'].dt.hour
filtered_data['Minute'] = filtered_data['Timestamp'].dt.minute

# NPK Ratio Class
# creating the NPKRatio since it isnt a column
# nitrogen phosphorus and potassium are input features
class NPKRatio:
    def __init__(self, phosphorous, nitrogen, potassium):
        self.phosphorous = phosphorous
        self.nitrogen = nitrogen
        self.potassium = potassium
    
    # a scaling method
    # ppm value is higher for the npk values to scale it down
    # if phosphorus == 0, it returns 0. we want it to scale it without returning a number hence this
    def calculate_ratio(self):
        if self.phosphorous == 0:
            return 0
        return (self.nitrogen / 4 + self.potassium / 1) / (self.phosphorous / 2)

# Calculate NPK Ratio
def calculate_npk_ratio(row):
    npk = NPKRatio(row['Phosphorous01'], row['Nitrogen01'], row['Potassium01'])
    return npk.calculate_ratio()

filtered_data['NPK_Ratio'] = filtered_data.apply(calculate_npk_ratio, axis=1)
print(filtered_data.columns)

# Anomaly Detection with Prophet
filtered_data_prophet = filtered_data[['Timestamp', 'NPK_Ratio']].rename(columns={'Timestamp': 'ds', 'NPK_Ratio': 'y'})
print(filtered_data_prophet.columns)
model = Prophet()
model.fit(filtered_data_prophet)
future = model.make_future_dataframe(periods=0)
forecast = model.predict(future)
filtered_data['yhat'] = forecast['yhat']
filtered_data['yhat_lower'] = forecast['yhat_lower']
filtered_data['yhat_upper'] = forecast['yhat_upper']
filtered_data['Anomaly'] = (filtered_data['NPK_Ratio'] < filtered_data['yhat_lower']) | (filtered_data['NPK_Ratio'] > filtered_data['yhat_upper'])
anomalies = filtered_data[filtered_data['Anomaly']]

# Plot Prophet results with anomalies
fig = model.plot(forecast)
# plt.scatter(anomalies['Timestamp'], anomalies['NPK_Ratio'], color='red')
# plt.show()

plt.scatter(anomalies['Timestamp'], anomalies['NPK_Ratio'], color='red')
plt.title('NPK Ratio Anomaly Detection')
plt.xlabel('Timestamp')
plt.ylabel('NPK Ratio')

# Save the plot as a PNG file
plt.savefig('output/anomaly_plot.png')

# Display the plot
plt.show()

# Set 'Timestamp' as index
filtered_data.set_index('Timestamp', inplace=True)

# Handle missing values by interpolation
filtered_data['NPK_Ratio'] = filtered_data['NPK_Ratio'].interpolate(method='linear')

# Perform time series decomposition
result = seasonal_decompose(filtered_data['NPK_Ratio'], model='additive', period=12)

# Plot the results with increased width
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 10))
result.observed.plot(ax=ax1, legend=False)
ax1.set_ylabel('Observed')
result.trend.plot(ax=ax2, legend=False)
ax2.set_ylabel('Trend')
result.seasonal.plot(ax=ax3, legend=False)
ax3.set_ylabel('Seasonal')
result.resid.plot(ax=ax4, legend=False)
ax4.set_ylabel('Residual')
plt.tight_layout()
plt.show()

filtered_data.head()

# Regression with Random Forest
features = filtered_data[['Phosphorous01', 'Nitrogen01', 'Potassium01', 'Temp01', 'Hum01', 'Heat01', 'SoilM01', 'Hour', 'Minute']]
target = filtered_data['NPK_Ratio']
features.fillna(features.mean(), inplace=True)
target.fillna(target.mean(), inplace=True)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print()
print(f"Random Forest Regressor: MSE = {mse:.2f}, R2 = {r2:.2f}")

# Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)
print("\nDecision Tree Regressor:")
print(f"MSE = {mse_dt:.2f}, R2 = {r2_dt:.2f}")

# Other Models
models = {'Ridge': Ridge(), 'Lasso': Lasso(), 'ElasticNet': ElasticNet()}
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MSE': mse, 'R2': r2}
for name, metrics in results.items():
    print(f"{name}: MSE = {metrics['MSE']:.2f}, R2 = {metrics['R2']:.2f}")