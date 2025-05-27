import pandas as pd
import matplotlib.pyplot as plt

# Load Excel file
df = pd.read_excel("section_article_scores_summary.xlsx")

# Convert columns to numeric, force errors to NaN
df['Mean'] = pd.to_numeric(df['Mean'], errors='coerce')
df['StdDev'] = pd.to_numeric(df['StdDev'], errors='coerce')

# Drop rows where either value is missing
df_clean = df.dropna(subset=['Mean', 'StdDev'])

# Extract data
all_means = df_clean['Mean'].values
all_sds = df_clean['StdDev'].values

# Plot histogram of Mean scores
plt.hist(all_means, bins=20, color='skyblue', edgecolor='black')
plt.title("Distribution of Mean Scores")
plt.xlabel("Mean")
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

# For Mean thresholding
Q1_mean = df_clean['Mean'].quantile(0.25)
Q3_mean = df_clean['Mean'].quantile(0.75)
IQR_mean = Q3_mean - Q1_mean
low_mean_threshold = max(0, Q1_mean - 1.5 * IQR_mean)  # truncate negative values to 0

# For SD thresholding
Q1_sd = df_clean['StdDev'].quantile(0.25)
Q3_sd = df_clean['StdDev'].quantile(0.75)
IQR_sd = Q3_sd - Q1_sd
high_sd_threshold = Q3_sd + 1.5 * IQR_sd

print(f"Low Mean Threshold: {low_mean_threshold:.2f}")
print(f"High SD Threshold: {high_sd_threshold:.2f}")

sorted_means = sorted(df_clean['Mean'])
plt.plot(sorted_means)
plt.title("Sorted Mean Scores")
plt.xlabel("Data Point Index")
plt.ylabel("Mean Score")
plt.grid(True)
plt.show()

low_mean_threshold = df_clean['Mean'].quantile(0.10)  # bottom 10%
high_sd_threshold = df_clean['StdDev'].quantile(0.90)  # top 10%

print(f"10th Percentile of Mean: {low_mean_threshold:.2f}")
print(f"90th Percentile of SD: {high_sd_threshold:.2f}")

