import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# create train df
train = pd.read_csv("data\\train.csv", index_col=0)

# create df of int features for plots
df_int = train.select_dtypes('float64')

# calculate rows dynamically
cols_per_row = 3
num_features = len(df_int.columns)
rows = int(np.ceil(len(df_int.columns) / 3))

# Boxplots for int features
plt.figure(figsize=(12,12))

n = 1
for column in df_int.columns:
  plt.subplot(rows,cols_per_row,n)
  n += 1
  sns.boxplot(x=train['Irrigation_Need'], y=train[f"{column}"], hue=train['Irrigation_Need'], legend=False)


plt.suptitle('Boxplots: Environmental Features vs. Irrigation Need', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.97]) 
#plt.show()
plt.savefig('plots\\int_boxplot.png')
plt.close()

# histogram for int features
plt.figure(figsize=(12,12))

n = 1
for column in df_int.columns:
  plt.subplot(rows,cols_per_row,n)
  n += 1
  sns.histplot(df_int[f"{column}"])


plt.suptitle('Histograms: Environmental Features', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
#plt.show()
plt.savefig('plots\\int_histogram.png')
plt.close()

# create df for obj features for plots
df_obj = train.select_dtypes('object')

# calculate rows dynamically
cols_per_row = 3
num_features = len(df_obj.columns)
rows = int(np.ceil(len(df_obj.columns) / 3))

# Boxplots for obj features
plt.figure(figsize=(12,12))

n = 1
for column in df_obj.columns:
  plt.subplot(rows,cols_per_row,n) # 4 rows, 3 columns
  n += 1
  sns.countplot(x = df_obj[f"{column}"], hue = df_obj['Irrigation_Need'])


plt.suptitle('Countplots: Environmental Features', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
#plt.show()
plt.savefig('plots\\obj_countplot.png')
plt.close()