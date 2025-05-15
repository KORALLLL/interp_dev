import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_DIR = r"C:\Users\Илья\Desktop\libritts\formants"


f1_list = []
f2_list = []
f3_list = []

for csv_filename in os.listdir(CSV_DIR):
    if csv_filename.endswith('.csv'):
        df = pd.read_csv(os.path.join(CSV_DIR, csv_filename))
        df_clean = df.dropna(subset=['F1', 'F2', 'F3'])

        f1_list.extend(df_clean['F1'].values)
        f2_list.extend(df_clean['F2'].values)
        f3_list.extend(df_clean['F3'].values)

f1_array = np.array(f1_list)
f2_array = np.array(f2_list)
f3_array = np.array(f3_list)

print(">>> Summary Statistics:")
print(f"F1 - Min: {np.min(f1_array)}, Max: {np.max(f1_array)}, Mean: {np.mean(f1_array)}, Std: {np.std(f1_array)}")
print(f"F2 - Min: {np.min(f2_array)}, Max: {np.max(f2_array)}, Mean: {np.mean(f2_array)}, Std: {np.std(f2_array)}")
print(f"F3 - Min: {np.min(f3_array)}, Max: {np.max(f3_array)}, Mean: {np.mean(f3_array)}, Std: {np.std(f3_array)}")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(f1_array, bins=50)
plt.title('F1 Distribution')
plt.xlabel('Hz')
plt.ylabel('Count')

plt.subplot(1, 3, 2)
plt.hist(f2_array, bins=50)
plt.title('F2 Distribution')
plt.xlabel('Hz')

plt.subplot(1, 3, 3)
plt.hist(f3_array, bins=50)
plt.title('F3 Distribution')
plt.xlabel('Hz')

plt.tight_layout()
plt.show()
