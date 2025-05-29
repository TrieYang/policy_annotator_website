import pandas as pd
import matplotlib.pyplot as plt

# Load Excel file
df = pd.read_excel("section_article_scores_summary.xlsx")

# Convert columns to numeric, force errors to NaN
df['Max'] = pd.to_numeric(df['Max'], errors='coerce')
df['StdDev'] = pd.to_numeric(df['StdDev'], errors='coerce')

# Drop rows where either value is missing
df_clean = df.dropna(subset=['Max', 'StdDev'])

# Extract data
all_maxs = df_clean['Max'].values
all_sds = df_clean['StdDev'].values

# Plot histogram of Max scores
plt.hist(all_maxs, bins=20, color='skyblue', edgecolor='black')
plt.title("Distribution of Max Scores")
plt.xlabel("Max")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Plot histogram of Standard Deviation scores
plt.hist(all_sds, bins=20, color='salmon', edgecolor='black')
plt.title("Distribution of Standard Deviations")
plt.xlabel("Standard Deviation")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()



sorted_maxs = sorted(df_clean['Max'])
plt.plot(sorted_maxs)
plt.title("Sorted Max Scores")
plt.xlabel("Data Point Index")
plt.ylabel("Max Score")
plt.grid(True)
plt.show()

low_max_threshold = df_clean['Max'].quantile(0.10)  # bottom 10%
high_sd_threshold = df_clean['StdDev'].quantile(0.90)  # top 10%

print(f"10th Percentile of Max: {low_max_threshold:.2f}")
print(f"90th Percentile of SD: {high_sd_threshold:.2f}")

