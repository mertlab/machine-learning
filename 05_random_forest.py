import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

melbourne_file_path = './archive/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 

# Filter rows with missing price values
filtered_melbourne_data = melbourne_data.dropna(axis=0)

# Choose target and features
y = filtered_melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']
X = filtered_melbourne_data[melbourne_features]

# split data into train and test
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# random forest regressor
forest_model = RandomForestRegressor(random_state=1)

# fit the model
forest_model.fit(train_X, train_y)

# predict
melb_preds = forest_model.predict(val_X)

# calculate mean absolute error
print(mean_absolute_error(val_y, melb_preds))