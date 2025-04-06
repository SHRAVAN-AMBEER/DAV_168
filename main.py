import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# LOAD THE DATASET
# -------------------------------
df = pd.read_csv("spacex_launch_data.csv")

# -------------------------------
# DATA CLEANING
# -------------------------------
df['Payload Mass (kg)'] = df['Payload Mass (kg)'].str.replace(' ', '', regex=False)
df['Payload Mass (kg)'] = pd.to_numeric(df['Payload Mass (kg)'], errors='coerce')
df['Flight Number'] = pd.to_numeric(df['Flight Number'], errors='coerce')
df['Payload Mass (kg)'] = df['Payload Mass (kg)'].fillna(df['Payload Mass (kg)'].mean())

# ==============================
# UNIT-I: NUMPY OPERATIONS
# ==============================
print("\n===== NUMPY OPERATIONS =====")

# 1. Fixed type arrays
arr = np.array([1, 2, 3, 4], dtype='int32')
print("\nFixed Type Array:", arr)

# 2. Creating arrays
zeros = np.zeros((2, 3))
ones = np.ones((2, 2))
rand = np.random.rand(2, 2)
print("\nZeros Array:\n", zeros)
print("\nOnes Array:\n", ones)
print("\nRandom Array:\n", rand)

# 3. Array indexing and slicing
a = np.arange(10)
print("\nOriginal Array:", a)
print("Sliced [2:7]:", a[2:7])
print("Fancy Indexing [1,3,5]:", a[[1, 3, 5]])

# 4. Reshaping arrays
reshaped = np.arange(12).reshape(3, 4)
print("\nReshaped (3x4):\n", reshaped)

# 5. Concatenation and splitting
concat = np.concatenate(([1, 2], [3, 4]))
split = np.split(concat, 2)
print("\nConcatenated:", concat)
print("Split:", split)

# 6. Universal functions
print("\nSquare root of arr:", np.sqrt(arr))

# 7. Aggregations
print("\nSum:", a.sum(), "Mean:", a.mean(), "Min:", a.min(), "Max:", a.max())

# 8. Broadcasting rules
print("\nBroadcasted arr + 5:", arr + 5)

# 9. Comparisons and Boolean arrays
print("\nComparison (arr > 2):", arr > 2)

# 10. Masks and filtering
mask = arr % 2 == 0
print("\nEven Mask:", mask)
print("Filtered Evens:", arr[mask])

# 11. Fancy indexing
print("\nFancy Indexing [2, 4, 6]:", a[[2, 4, 6]])

# 12. Fast sorting
print("\nSorted:", np.sort(arr))
print("Argsort (indices):", np.argsort(arr))
print("Partial sort (partition at 3):", np.partition(arr, 3))

# 13. Structured arrays, Compound and Record types
data = np.array([('Falcon 1', 1), ('Falcon 9', 2)], dtype=[('name', 'U10'), ('version', 'i4')])
record = data.view(np.recarray)
print("\nStructured Array (Record Type):\n", record)

# ==============================
# UNIT-II: PANDAS OPERATIONS
# ==============================
print("\n===== PANDAS OPERATIONS =====")

# 1. Series object
s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
print("\nSeries:\n", s)

# 2. DataFrame object
df_series = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
print("\nDataFrame:\n", df_series)

# 3. Data Indexing and Selecting
print("\nFirst 5 Payloads:\n", df['Payload'][:5])
print("\nSelected Columns:\n", df[['Flight Number', 'Mission Outcome']].head())

# 4. Universal Functions and Index Preservation
df['Flight Number Squared'] = df['Flight Number'] ** 2
df['Double FN'] = df['Flight Number'] + df['Flight Number']
print("\nFlight Number Squared & Doubled:\n", df[['Flight Number', 'Flight Number Squared', 'Double FN']].head())

# 5. Handling Missing Data
print("\nMissing Values:\n", df.isnull().sum())

# 6. Hierarchical Indexing
df_hierarchical = df.set_index(['Launch Site', 'Orbit'])
print("\nHierarchical Index Sample:\n", df_hierarchical.head())

# ==============================
# UNIT-III: COMBINING DATASETS
# ==============================
print("\n===== COMBINING DATASETS =====")

# 1. Concat
df_concat = pd.concat([df.head(2), df.tail(2)])
print("\nConcatenated Rows:\n", df_concat)

# 2. Append
df_appended = pd.concat([df.head(2), df.tail(2)], ignore_index=True)
print("\nAppended Data:\n", df_appended)

# 3. Merge and Joins
meta = pd.DataFrame({'Flight Number': [1, 2], 'Success Rating': ['Low', 'Medium']})
df_merged = pd.merge(df, meta, on='Flight Number', how='left')
print("\nMerged with Meta:\n", df_merged[['Flight Number', 'Success Rating']].dropna().head())

# 4. Aggregation and Grouping
grouped = df.groupby('Launch Site')['Payload Mass (kg)'].mean()
print("\nGrouped Mean Payload per Site:\n", grouped)

# 5. Pivot Table
pivot = df.pivot_table(values='Payload Mass (kg)', index='Launch Site', columns='Orbit', aggfunc='mean')
print("\nPivot Table:\n", pivot)

# ==============================
# GRAPHS (DISPLAYED INLINE)
# ==============================

# 1. Histogram - Payload Mass
plt.figure(figsize=(10, 6))
plt.hist(df['Payload Mass (kg)'].dropna(), bins=30, color='skyblue', edgecolor='black')
plt.title('Payload Mass Distribution')
plt.xlabel('Payload Mass (kg)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 2. Bar Chart - Launch Count per Site
launch_counts = df['Launch Site'].value_counts()
plt.figure(figsize=(12, 6))
plt.bar(launch_counts.index, launch_counts.values, color='orange', edgecolor='black')
plt.title('Number of Launches per Site')
plt.xlabel('Launch Site')
plt.ylabel('Number of Launches')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Box Plot - Payload Mass by Mission Outcome
plt.figure(figsize=(12, 6))
outcomes = df['Mission Outcome'].unique()
data_to_plot = [df[df['Mission Outcome'] == outcome]['Payload Mass (kg)'].dropna() for outcome in outcomes]
plt.boxplot(data_to_plot, tick_labels=outcomes, patch_artist=True)
plt.title('Payload Mass by Mission Outcome')
plt.xlabel('Mission Outcome')
plt.ylabel('Payload Mass (kg)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
