import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Generate Simulated Crime Dataset
np.random.seed(123)

crime_data = pd.DataFrame({
    "Report_Number": range(1, 1001),
    "Date_Reported": np.random.choice(pd.date_range(start="2020-01-01", end="2024-01-01"), 1000),
    "Date_Occurred": np.random.choice(pd.date_range(start="2020-01-01", end="2024-01-01"), 1000),
    "Time_Occurred": np.random.choice(pd.date_range("00:00", "23:59", freq="H").time, 1000),
    "City": np.random.choice(["Mumbai", "Delhi", "Bangalore", "Kolkata", "Chennai", "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow"], 1000),
    "Crime_Code": np.random.randint(100, 999, 1000),
    "Crime_Description": np.random.choice(["Theft", "Assault", "Robbery", "Murder", "Burglary", "Vandalism", "Fraud", "Sexual Assault"], 1000),
    "Victim_Age": np.random.randint(18, 80, 1000),
    "Victim_Gender": np.random.choice(["Male", "Female"], 1000),
    "Weapon_Used": np.random.choice(["None", "Knife", "Gun", "Bat", "Other"], 1000),
    "Crime_Domain": np.random.choice(["Property Crime", "Violent Crime", "White-Collar Crime", "Drug-Related Crime"], 1000),
    "Police_Deployed": np.random.choice(["Yes", "No"], 1000),
    "Case_Closed": np.random.choice(["Yes", "No"], 1000),
    "Date_Case_Closed": np.random.choice(pd.date_range(start="2020-01-01", end="2024-01-01"), 1000)
})

# Convert Time_Occurred to Hour for analysis
crime_data["Hour_Occurred"] = pd.to_datetime(crime_data["Time_Occurred"], format="%H:%M:%S").dt.hour

# Summary Statistics
summary_stats = crime_data.groupby(["City", "Crime_Description"]).agg(
    Total_Crimes=("Crime_Description", "count"),
    Average_Victim_Age=("Victim_Age", "mean"),
    Case_Closed_Rate=("Case_Closed", lambda x: (x == "Yes").mean())
).reset_index()

# -------------------- 1. Crime Distribution by City (Stacked Bar Plot) --------------------
plt.figure(figsize=(12, 6))
sns.countplot(data=crime_data, x="City", hue="Crime_Description")
plt.xticks(rotation=45)
plt.title("Crime Distribution by City")
plt.xlabel("City")
plt.ylabel("Number of Crimes")
plt.legend(title="Crime Type")
plt.show()

# -------------------- 2. Victim Age Distribution by Crime Type (Density Plot) --------------------
plt.figure(figsize=(10, 6))
for crime_type in crime_data["Crime_Description"].unique():
    sns.kdeplot(crime_data[crime_data["Crime_Description"] == crime_type]["Victim_Age"], label=crime_type, fill=True, alpha=0.4)
plt.title("Victim Age Distribution by Crime Type")
plt.xlabel("Age")
plt.ylabel("Density")
plt.legend()
plt.show()

# -------------------- 3. Time of Crime Occurrence (Hourly Histogram) --------------------
plt.figure(figsize=(10, 6))
sns.histplot(data=crime_data, x="Hour_Occurred", hue="Crime_Description", multiple="stack", bins=24)
plt.title("Time of Crime Occurrence (Hourly)")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Crimes")
plt.show()

# -------------------- 4. Crime Closure Rate by City (Bar Plot) --------------------
plt.figure(figsize=(12, 6))
sns.barplot(data=summary_stats, x="City", y="Case_Closed_Rate", hue="Crime_Description")
plt.xticks(rotation=45)
plt.title("Crime Closure Rate by City")
plt.xlabel("City")
plt.ylabel("Closure Rate")
plt.legend(title="Crime Type")
plt.show()

# -------------------- 5. Victim Age Distribution by Crime Domain (Box Plot) --------------------
plt.figure(figsize=(12, 6))
sns.boxplot(data=crime_data, x="Crime_Domain", y="Victim_Age", palette="coolwarm")
plt.title("Victim Age Distribution by Crime Domain")
plt.xlabel("Crime Domain")
plt.ylabel("Victim Age")
plt.show()

# -------------------- 6. Scatter Plot for Victim Age vs. Time of Occurrence --------------------
plt.figure(figsize=(10, 6))
sns.scatterplot(data=crime_data, x="Hour_Occurred", y="Victim_Age", hue="Crime_Description", alpha=0.7)
plt.title("Victim Age vs. Time of Occurrence")
plt.xlabel("Hour of Day")
plt.ylabel("Victim Age")
plt.legend(title="Crime Type")
plt.show()

# -------------------- 7. Crime Rate Over Time (Yearly) --------------------
crime_data["Year"] = pd.to_datetime(crime_data["Date_Occurred"]).dt.year

plt.figure(figsize=(10, 6))
sns.countplot(data=crime_data, x="Year", hue="Crime_Description")
plt.title("Crime Rate Over Time (Yearly)")
plt.xlabel("Year")
plt.ylabel("Number of Crimes")
plt.legend(title="Crime Type")
plt.show()

# -------------------- 8. Heatmap of Crime Occurrences by City and Crime Type --------------------
crime_count_matrix = crime_data.groupby(["City", "Crime_Description"]).size().unstack(fill_value=0)

plt.figure(figsize=(12, 6))
sns.heatmap(crime_count_matrix, cmap="Reds", annot=True, fmt="d", linewidths=0.5)
plt.title("Heatmap of Crime Occurrences by City and Crime Type")
plt.xlabel("Crime Type")
plt.ylabel("City")
plt.xticks(rotation=45)
plt.show()
