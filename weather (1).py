# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load Dataset
df = pd.read_csv("weather_data.csv")

# Display first 5 rows
print("Dataset Preview:")
print(df.head())

# Step 2: Data Preprocessing
print("\nMissing Values:")
print(df.isnull().sum())

# Remove missing values
df = df.dropna()

# Step 3: Feature Engineering
# Create Season column
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8]:
        return "Monsoon"
    else:
        return "Autumn"

df["Season"] = df["Month"].apply(get_season)

print("\nDataset with Season Column:")
print(df.head())

# Step 4: Descriptive Statistics
print("\nStatistical Summary:")
print(df.describe())

# Step 5: Data Analysis

# Average Temperature by Season
print("\nAverage Temperature by Season:")
temp_season = df.groupby("Season")["Temperature"].mean()
print(temp_season)

# Average Rainfall by Season
print("\nAverage Rainfall by Season:")
rain_season = df.groupby("Season")["Rainfall"].mean()
print(rain_season)

# Step 6: Correlation Analysis
print("\nCorrelation Matrix:")
corr = df.corr(numeric_only=True)
print(corr)

# ----------- DATA VISUALIZATION -----------

# 1. Temperature by Season
plt.figure()
temp_season.plot(kind="bar")
plt.title("Average Temperature by Season")
plt.xlabel("Season")
plt.ylabel("Temperature")

# 2. Rainfall by Season
plt.figure()
rain_season.plot(kind="bar")
plt.title("Average Rainfall by Season")
plt.xlabel("Season")
plt.ylabel("Rainfall")

# 3. Rainfall vs Humidity
plt.figure()
plt.scatter(df["Rainfall"], df["Humidity"])
plt.title("Rainfall vs Humidity")
plt.xlabel("Rainfall")
plt.ylabel("Humidity")

# 4. Temperature vs Wind Speed
plt.figure()
plt.scatter(df["Temperature"], df["WindSpeed"])
plt.title("Temperature vs Wind Speed")
plt.xlabel("Temperature")
plt.ylabel("Wind Speed")

# 5. Temperature Distribution
plt.figure()
plt.hist(df["Temperature"])
plt.title("Temperature Distribution")
plt.xlabel("Temperature")
plt.ylabel("Frequency")

# 6. Correlation Heatmap
plt.figure()
plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Heatmap")

# Show all graphs
plt.show()

# Final Message
print("\nAnalysis Completed Successfully!")