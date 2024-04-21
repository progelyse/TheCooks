import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Define your CSV path
csvpath = "C:/Users/user/Downloads/ibmchal/"

# Load the data
PLANE_DF = pd.read_csv(csvpath+'plane.csv', low_memory=False)
TAXI_DF = pd.read_csv(csvpath+'taxi.csv', low_memory=False)

# Define center latitude, longitude, and radius for the airport
center_latitude = 41.98
center_longitude = -87.9
radius = 0.141

# Calculate squared distance from the airport
TAXI_DF['distance_squared'] = ((TAXI_DF['pickup_latitude'] - center_latitude)**2 +
                               (TAXI_DF['pickup_longitude'] - center_longitude)**2)

# Filter out trips that are within the specified radius
airport_related_trips = TAXI_DF[TAXI_DF['distance_squared'] <= radius**2].copy()

# Preprocess Taxi data for trips within the radius
airport_related_trips['trip_start_timestamp'] = pd.to_datetime(airport_related_trips['trip_start_timestamp'])
airport_related_trips['year'] = airport_related_trips['trip_start_timestamp'].dt.year
airport_related_trips['month'] = airport_related_trips['trip_start_timestamp'].dt.month
airport_related_trips['day'] = airport_related_trips['trip_start_timestamp'].dt.day
airport_related_trips['hour'] = airport_related_trips['trip_start_timestamp'].dt.hour
grouped_taxi_df = airport_related_trips.groupby(['year', 'month', 'day', 'hour']).size().reset_index(name='trip_count')

# Preprocess Plane data
PLANE_DF['arrtime_str'] = PLANE_DF['arrtime'].apply(lambda x: '{0:0>4}'.format(x))
PLANE_DF['arrival_hour'] = PLANE_DF['arrtime_str'].str[:2].astype(int)
PLANE_DF['year'] = PLANE_DF['year'].astype(int)
PLANE_DF['month'] = PLANE_DF['month'].astype(int)
PLANE_DF['dayofmonth'] = PLANE_DF['dayofmonth'].astype(int)
grouped_plane_df = PLANE_DF.groupby(['year', 'month', 'dayofmonth', 'arrival_hour']).size().reset_index(name='flight_count')

# Rename columns for merging consistency
grouped_plane_df.rename(columns={'dayofmonth': 'day', 'arrival_hour': 'hour'}, inplace=True)

# Merge the plane and taxi data on common columns
merged_df = pd.merge(grouped_plane_df, grouped_taxi_df, on=['year', 'month', 'day', 'hour'], how='inner')

# Prepare data for model training
X = merged_df[['flight_count', 'hour']]
y = merged_df['trip_count']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Regressor
dt_regressor = DecisionTreeRegressor(random_state=42)
dt_regressor.fit(X_train, y_train)

# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Predict on the test set using both models
dt_y_pred = dt_regressor.predict(X_test)
rf_y_pred = rf_regressor.predict(X_test)

# Evaluate the Decision Tree model
dt_mse = mean_squared_error(y_test, dt_y_pred)
dt_mae = mean_absolute_error(y_test, dt_y_pred)
dt_rmse = np.sqrt(mean_squared_error(y_test, dt_y_pred))

# Evaluate the Random Forest model
rf_mse = mean_squared_error(y_test, rf_y_pred)
rf_mae = mean_absolute_error(y_test, rf_y_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_y_pred))

# Print the results
print("Decision Tree Mean Squared Error:", dt_mse)
print("Decision Tree Mean Absolute Error:", dt_mae)
print("Decision Tree Root Mean Squared Error:", dt_rmse)
print("Random Forest Mean Squared Error:", rf_mse)
print("Random Forest Mean Absolute Error:", rf_mae)
print("Random Forest Root Mean Squared Error:", rf_rmse)

# Display predictions from the Random Forest model
rf_results = pd.DataFrame({'Actual': y_test, 'Predicted': rf_y_pred})
print(rf_results.head())

# Ask user for the hour to predict taxi demand
hour_to_predict = int(input("Input an hour to predict taxi demand (0-23): "))

# Average flight count for the hour entered by the user
# Replace this with your actual method to get the average flight count for a given hour
avg_flight_count = merged_df[merged_df['hour'] == hour_to_predict]['flight_count'].mean()

# Prepare the single input for prediction
input_features = np.array([[avg_flight_count, hour_to_predict]])

# Predict the taxi demand for the given hour
predicted_taxi_demand = rf_regressor.predict(input_features)

# Output the prediction result
print(f"Predicted number of taxis needed at hour {hour_to_predict}: {int(predicted_taxi_demand[0])}")

# Plotting flight counts vs taxi trips
plt.figure(figsize=(12, 6))
sns.lineplot(data=merged_df, x='hour', y='trip_count', label='Taxi Trips')
sns.lineplot(data=merged_df, x='hour', y='flight_count', label='Flight Arrivals')
plt.title('Comparison of Flight Arrivals and Taxi Trips by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Count')
plt.legend()
plt.show()
