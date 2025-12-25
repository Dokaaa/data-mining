import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://raw.githubusercontent.com/mlco2/impact/master/data/instances.csv"
df = pd.read_csv(url)

print(df.head())
print(df.isnull().sum())

df_clean = df.dropna(subset=['gpu', 'provider'])
print(f"Number of valid rows: {len(df_clean)}")

gpu_tdp = {
    'Tesla V100': 250,
    'Tesla K80': 150,
    'A100': 300,
    'T4': 70,
    'P100': 250,
    'RTX 3090': 350,
    'RTX 4090': 450
}

df_clean['power_draw'] = df_clean['gpu'].map(gpu_tdp).fillna(200)  # неизвестные GPU = 200W

gpu_counts = df_clean['gpu'].value_counts()
plt.figure(figsize=(8,5))
gpu_counts.plot(kind='bar', color='orange')
plt.xlabel("GPU Type")
plt.ylabel("Number of Instances")
plt.title("Frequency of GPU Types in Dataset")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

gpu_energy = df_clean.groupby('gpu')['power_draw'].mean().sort_values(ascending=False)
plt.figure(figsize=(8,5))
gpu_energy.plot(kind='bar', color='red')
plt.xlabel("GPU Type")
plt.ylabel("Average Power Draw (W)")
plt.title("Average Energy Consumption by GPU")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

provider_energy = df_clean.groupby('provider')['power_draw'].sum().sort_values(ascending=False)
plt.figure(figsize=(6,4))
provider_energy.plot(kind='bar', color='green')
plt.xlabel("Cloud Provider")
plt.ylabel("Total Power Draw (W)")
plt.title("Total Energy Consumption per Provider")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

pivot = df_clean.pivot_table(index='gpu', columns='provider', values='power_draw', aggfunc='sum', fill_value=0)
plt.figure(figsize=(10,6))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title("Total GPU Power Draw per Provider (W)")
plt.ylabel("GPU Type")
plt.xlabel("Provider")
plt.show()

plt.figure(figsize=(8,5))
plt.hist(df_clean['power_draw'], bins=6, color='blue', edgecolor='black')
plt.xlabel("Power Draw (W)")
plt.ylabel("Number of Instances")
plt.title("Distribution of GPU Power Draw")
plt.show()

df_clean['co2_t'] = df_clean['power_draw'] * 1 / 1000 * 0.0004  # W*h → kWh → tCO2

plt.figure(figsize=(8,5))
plt.scatter(df_clean['gpu'], df_clean['co2_t'], color='purple', s=100)
plt.xlabel("GPU Type")
plt.ylabel("Estimated CO2 Emissions (t)")
plt.title("Estimated CO2 Emissions per GPU Instance")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()