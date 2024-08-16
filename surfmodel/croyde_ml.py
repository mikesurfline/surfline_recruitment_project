import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Let's load the data
df = pd.read_csv("Croyde.csv")

# Convert surf model output to FT
df["SURFMODEL_MAXBWH_FT"] = df["SURFMODEL_MAXBWH_MT"] * 3.2808

# Get rid of rows with missing observations
df = df.dropna(subset=['OBSERVATION_MAXBWH_FT'])

# split dataframe into 80% training data and 20% test data
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# Prepare labels (i.e. observations)
y_train = df_train['OBSERVATION_MAXBWH_FT']
y_val = df_test['OBSERVATION_MAXBWH_FT']

#First, let's build a linear regression model from the SURFMODEL data
X_train_surf = df_train[['SURFMODEL_MAXBWH_FT']]
X_val_surf = df_test[['SURFMODEL_MAXBWH_FT']]
# Train Linear Regression on SURFMODEL_MAXBWH_FT
lr_model = LinearRegression(fit_intercept=False)
lr_model.fit(X_train_surf, y_train)
# With Linear Regression, we can directly get the coefficients and intercept as it's just a*x + b
print(lr_model.coef_)  # that's the slope
print(lr_model.intercept_)  # that's the intercept - it's 0 because we set fit_intercept=False above

#Now to a more advanced model, let's build a Random Forest Regressor model from the WAVEMODEL data
# Prepare input data (i.e. WAVEMODEL_XXX data)
wave_cols = [col for col in df_train.columns if col.startswith('WAVE')]
X_train_wave = df_train[wave_cols]
X_val_wave = df_test[wave_cols]
# Train Random Forest on WAVE features
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_wave, y_train)

# Make predictions on the test dataframe
y_pred_rf = rf_model.predict(X_val_wave)
y_pred_lr = lr_model.predict(X_val_surf)

# Add predictions to the test dataframe
df_test['PREDICTED_LR_MAXBWH_FT'] = y_pred_lr
df_test['PREDICTED_RF_MAXBWH_FT'] = y_pred_rf
