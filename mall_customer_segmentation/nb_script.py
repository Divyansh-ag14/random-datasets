# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df = pd.read_csv("./Mall_Customers.csv")

# %%
df.shape

# %%
df.head()

# %%
df.isna().sum()

# %%
df.dtypes

# %%
df.describe().T

# %%
df.drop("CustomerID", axis=1, inplace=True)

# %%
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
df["Gender"] = enc.fit_transform(df["Gender"])

# %%
enc.classes_

# %%
df.head()

# %%
gender_mappings = {index: label for index, label in enumerate(enc.classes_)}
gender_mappings

# %%
from sklearn.preprocessing import StandardScaler

# %%
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# %%
df_scaled.head()

# %%
from sklearn.cluster import KMeans

# %%
clusters = 100

# %%
kmeans_tests = [KMeans(n_clusters=i, n_init=10) for i in range(1, clusters)]
inertias = [kmeans_tests[i].fit(df_scaled).inertia_ for i in range(len(kmeans_tests))]

# %%
plt.figure(figsize=(7, 5))
plt.plot(range(1, clusters), inertias)
plt.xlabel("Number of Clusters")
plt.xticks(np.arange(0, 99, 10))
plt.ylabel("Inertia")
plt.title("Choosing the Number of Clusters")
plt.show()

# %%
model = KMeans(n_clusters=10, n_init=10)
model.fit(df_scaled)

# %%
clusters = model.predict(df_scaled)
clusters

# %%
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
reduced_data = pd.DataFrame(pca.fit_transform(df_scaled), columns=['PC1', 'PC2'])

# %%
reduced_data.head()

# %%
model.cluster_centers_

# %%
new_centers = pca.transform(model.cluster_centers_)
new_centers

# %%
reduced_data['cluster'] = clusters

# %%
reduced_data.head()

# %%
plt.figure(figsize=(14, 10))

plt.scatter(reduced_data[reduced_data['cluster'] == 0].loc[:, 'PC1'], reduced_data[reduced_data['cluster'] == 0].loc[:, 'PC2'], color='red')
plt.scatter(reduced_data[reduced_data['cluster'] == 1].loc[:, 'PC1'], reduced_data[reduced_data['cluster'] == 1].loc[:, 'PC2'], color='blue')
plt.scatter(reduced_data[reduced_data['cluster'] == 2].loc[:, 'PC1'], reduced_data[reduced_data['cluster'] == 2].loc[:, 'PC2'], color='yellow')
plt.scatter(reduced_data[reduced_data['cluster'] == 3].loc[:, 'PC1'], reduced_data[reduced_data['cluster'] == 3].loc[:, 'PC2'], color='orange')
plt.scatter(reduced_data[reduced_data['cluster'] == 4].loc[:, 'PC1'], reduced_data[reduced_data['cluster'] == 4].loc[:, 'PC2'], color='cyan')
plt.scatter(reduced_data[reduced_data['cluster'] == 5].loc[:, 'PC1'], reduced_data[reduced_data['cluster'] == 5].loc[:, 'PC2'], color='magenta')
plt.scatter(reduced_data[reduced_data['cluster'] == 6].loc[:, 'PC1'], reduced_data[reduced_data['cluster'] == 6].loc[:, 'PC2'], color='brown')
plt.scatter(reduced_data[reduced_data['cluster'] == 7].loc[:, 'PC1'], reduced_data[reduced_data['cluster'] == 7].loc[:, 'PC2'], color='pink')
plt.scatter(reduced_data[reduced_data['cluster'] == 8].loc[:, 'PC1'], reduced_data[reduced_data['cluster'] == 8].loc[:, 'PC2'], color='green')
plt.scatter(reduced_data[reduced_data['cluster'] == 9].loc[:, 'PC1'], reduced_data[reduced_data['cluster'] == 9].loc[:, 'PC2'], color='purple')

plt.scatter(new_centers[:, 0], new_centers[:, 1], color='black', marker='x', s=300)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.show()

# %%
sns.set_style("darkgrid")
colors = ["red", "blue", "yellow", "orange", "cyan", "magenta", "brown", "pink", "green", "purple"]
plt.figure(figsize=(14, 10))
for i, color in enumerate(colors):
    plt.scatter(reduced_data[reduced_data['cluster'] == i].loc[:, 'PC1'], reduced_data[reduced_data['cluster'] == i].loc[:, 'PC2'], color=color)

plt.scatter(new_centers[:, 0], new_centers[:, 1], color='black', marker='x', s=300)

plt.xlabel("PC1")
plt.ylabel("PC2")

# %%



