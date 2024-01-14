import seaborn as sns
import pandas as pd
import math

iris_df=pd.read_csv("/config/workspace/Assignment01/Iris.csv")
iris_df=iris_df.drop('Id',axis=1)
# Find the mean and median of the 'sepal_length' column.
sepal_l_mean=iris_df['SepalLengthCm'].mean()
sepal_l_median=iris_df['SepalLengthCm'].median()
print(f"Mean of sepal length: {sepal_l_mean}")
print(f"Median of sepal length: {sepal_l_median}")

# Calculate the 75th percentile of the 'petal_width' column for each species in the Iris dataset.
petal_width_percentile_75=iris_df.groupby('Species')['PetalWidthCm'].quantile(0.75)
print(f"75th percentile of the 'petal_width' for each species: {petal_width_percentile_75} ")

# Create a new column in the Iris DataFrame called 'sepal_area', which is the product of 'sepal_length' and 'sepal_width'.
iris_df["sepal_area"]=iris_df['SepalLengthCm']*iris_df['SepalWidthCm']
print(iris_df.head(5))

# Remove all rows in the Iris DataFrame where 'petal_length' is greater than twice the standard deviation of 'petal_length' for that species.
def remove_outliers_by_Species(group):
    return group[abs(group['PetalLengthCm']-group['PetalLengthCm'].mean())<=2*group['PetalLengthCm'].std()]

iris_df=iris_df.groupby('Species').apply(remove_outliers_by_Species).reset_index(drop=True)
print(iris_df)

# Normalize all numerical columns in the Iris DataFrame (except the 'species' column) using Min-Max scaling.
numeric_col=iris_df.select_dtypes(include='number').columns
iris_df[numeric_col]=(iris_df[numeric_col]-iris_df[numeric_col].min())/(iris_df[numeric_col].max()-iris_df[numeric_col].min())
print(iris_df)

#  Find the three most common combinations of 'sepal_length', 'sepal_width', and 'petal_length' in the Iris dataset.
most_common_comb=iris_df.groupby(['SepalLengthCm','SepalWidthCm','PetalLengthCm']).size().nlargest(3)
print(f"three most common combinations of 'sepal_length', 'sepal_width', and 'petal_length': {most_common_comb}")

# Group the Iris DataFrame by 'species' and find the row with the highest 'sepal_width' for each group.
max_sepal_width_rw=iris_df.loc[iris_df.groupby('Species')['SepalWidthCm'].idxmax()]
print(f"row with the highest 'sepal_width' for each group: {max_sepal_width_rw}")

# Replace all negative values in the 'petal_width' column of the Iris DataFrame with the mean of the non-negative values in that column.
non_neg_petal_witdth=iris_df['PetalWidthCm'].replace(iris_df[iris_df['PetalWidthCm']>=0]['PetalWidthCm'].mean())
print(f"Replace all negative values in the 'petal_width': {non_neg_petal_witdth}")

# Calculate the correlation matrix for the 'sepal_length', 'sepal_width', 'petal_length', and 'petal_width' columns in the Iris dataset and find the feature with the highest absolute correlation with 'petal_width'.
corr_mat=iris_df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].corr()
highest_corr=feature=corr_mat['PetalWidthCm'].abs().idxmax()
print(f"correlation matrix:\n {corr_mat}")
print("\n")
print(f"feature with the highest absolute correlation with 'petal_width': {highest_corr}")
