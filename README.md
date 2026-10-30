# ML and AML Practicals --- Selected Questions

Solutions are based on **AML ML Practical Questions (1).pdf**. Each
section is a standalone example. Install the required libraries first:

``` bash
pip install numpy pandas matplotlib scikit-learn
```

For CSV-based questions, place the relevant CSV file in the same folder
as the Python script. Check column names in your file and update them in
the code if necessary.

------------------------------------------------------------------------

## Q1. Decision Tree Classifier --- Diabetes Dataset

**Aim:** Train a decision tree using the complete `diabetes.csv`
dataset, use all features except `label`, limit depth to 3, and plot the
tree.

``` python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

df = pd.read_csv("diabetes.csv")
X = df.drop(columns=["label"])
y = df["label"]

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

print("Training accuracy:", accuracy_score(y, model.predict(X)))

plt.figure(figsize=(16, 8))
plot_tree(model, feature_names=X.columns, class_names=True,
          filled=True, rounded=True)
plt.title("Diabetes Decision Tree (Maximum Depth = 3)")
plt.show()
```

## Q3. Compare Decision Tree and SVM on Iris

``` python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(kernel="rbf")
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name} accuracy: {accuracy_score(y_test, model.predict(X_test)):.3f}")
```

## Q4. Iris Decision Trees --- `max_leaf_nodes=8` vs `min_samples_leaf=5`

``` python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "max_leaf_nodes=8": DecisionTreeClassifier(max_leaf_nodes=8, random_state=42),
    "min_samples_leaf=5": DecisionTreeClassifier(min_samples_leaf=5, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name} accuracy: {accuracy_score(y_test, model.predict(X_test)):.3f}")
```

## Q8. Elbow Method on 14 Generated Points

``` python
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=14, centers=3, n_features=2, random_state=42)
inertias = []

for k in range(1, 8):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)
    inertias.append(model.inertia_)

plt.plot(range(1, 8), inertias, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.xticks(range(1, 8))
plt.grid(True)
plt.show()
```

## Q12. K-Means on 300 `make_blobs` Samples --- Compare 2 to 7 Clusters

``` python
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=300, centers=4, n_features=2,
                  cluster_std=1.2, random_state=42)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for k, ax in zip(range(2, 8), axes.ravel()):
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=25)
    ax.set_title(f"K-Means: k={k}")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

plt.tight_layout()
plt.show()
```

## Q14. K-Means on 35 Random Points --- Impact of Cluster Count

``` python
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=35, centers=4, n_features=2, random_state=42)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for k, ax in zip([2, 3, 5], axes):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=45)
    ax.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
               marker="X", s=180, edgecolor="black")
    ax.set_title(f"k = {k}")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

plt.tight_layout()
plt.show()
```

**Impact:** Smaller `k` may combine distinct groups; larger `k` divides
data into more, smaller clusters. Select `k` based on the plot and
clustering evaluation, not just appearance.

## Q15. DBSCAN Clustering Using `make_moons`

``` python
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

X, _ = make_moons(n_samples=300, noise=0.08, random_state=42)
X = StandardScaler().fit_transform(X)

labels = DBSCAN(eps=0.25, min_samples=5).fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=25)
plt.title("DBSCAN on make_moons (-1 means noise)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

print("Cluster labels:", set(labels))
print("Noise points:", sum(labels == -1))
```

## Q27. PCA on Iris --- Reduce 4 Features to 2

``` python
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

iris = load_iris()
X, y = iris.data, iris.target
X_scaled = StandardScaler().fit_transform(X)

X_pca = PCA(n_components=2).fit_transform(X_scaled)

for i, name in enumerate(iris.target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=name)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA on Iris Dataset")
plt.legend()
plt.show()
```

## Q33. PCA on Iris --- Explore Data and Plot 2D Results

``` python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = [iris.target_names[i] for i in iris.target]

print("Dataset shape:", df.shape)
print(df.head())

X = StandardScaler().fit_transform(iris.data)
X_pca = PCA(n_components=2).fit_transform(X)

for i, name in enumerate(iris.target_names):
    plt.scatter(X_pca[iris.target == i, 0],
                X_pca[iris.target == i, 1], label=name)

plt.title("Iris Dataset After PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()
```

## Q34. LDA on Iris --- Reduce to 2 Dimensions

``` python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = [iris.target_names[i] for i in iris.target]
print("Dataset shape:", df.shape)
print(df.head())

X = StandardScaler().fit_transform(iris.data)
y = iris.target
X_lda = LinearDiscriminantAnalysis(n_components=2).fit_transform(X, y)

for i, name in enumerate(iris.target_names):
    plt.scatter(X_lda[y == i, 0], X_lda[y == i, 1], label=name)

plt.title("LDA on Iris Dataset")
plt.xlabel("Linear Discriminant 1")
plt.ylabel("Linear Discriminant 2")
plt.legend()
plt.show()
```

## Q36. LDA on Iris with Label Encoding and 2D Visualization

The target labels from `load_iris()` are already encoded as integers.
`LabelEncoder` is included to demonstrate the requested preprocessing.

``` python
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris = load_iris()
X = StandardScaler().fit_transform(iris.data)
labels = [iris.target_names[i] for i in iris.target]
y = LabelEncoder().fit_transform(labels)

X_lda = LinearDiscriminantAnalysis(n_components=2).fit_transform(X, y)

for i, name in enumerate(iris.target_names):
    plt.scatter(X_lda[y == i, 0], X_lda[y == i, 1], label=name)

plt.title("LDA on Iris with Label Encoding")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.legend()
plt.show()
```

## Q37. LDA on Breast Cancer Dataset --- Reduce to 1 Dimension

``` python
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

data = load_breast_cancer()
X = StandardScaler().fit_transform(data.data)
y = data.target

X_lda = LinearDiscriminantAnalysis(n_components=1).fit_transform(X, y)

plt.figure(figsize=(9, 4))
for cls, name in enumerate(data.target_names):
    plt.scatter(X_lda[y == cls, 0], [cls] * sum(y == cls),
                label=name, alpha=0.6)

plt.title("Breast Cancer Dataset: 1D LDA")
plt.xlabel("Linear Discriminant 1")
plt.yticks([0, 1], data.target_names)
plt.legend()
plt.show()
```

## Q44. Gaussian Naive Bayes --- Breast Cancer Dataset

``` python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=data.target_names))
```

## Q45. Gaussian Naive Bayes --- PIMA Diabetes Dataset

Assumes the target column is named `Outcome`. If your file uses
`Diabetes` or another name, change `TARGET`.

``` python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("PIMA.csv")
TARGET = "Outcome"  # Change this to your actual target column name

print(df.head())
print(df.describe())
print("Missing values:\n", df.isnull().sum())

df = df.dropna()
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

**Bias note:** The dataset represents Pima Indian women aged 21 and
older, so results may not generalize to other populations. Small or
unrepresentative samples, missing values, and zero values used to
represent missing medical measurements can affect predictions.

## Q48. Logistic Regression --- Customer Churn (Given 10 Records)

``` python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.DataFrame({
    "Age": [25, 34, 45, 29, 50, 38, 42, 35, 48, 55],
    "Account_Length": [12, 24, 36, 18, 48, 30, 42, 24, 36, 60],
    "Monthly_Charges": [70, 90, 80, 100, 60, 80, 90, 70, 80, 100],
    "Churn": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
})

X = df[["Age", "Account_Length", "Monthly_Charges"]]
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Predicted churn:", pred)
print("Churn probabilities:\n", model.predict_proba(X_test))
print("Accuracy:", accuracy_score(y_test, pred))
```

**Note:** This is a very small 10-row example, so its accuracy is not a
reliable estimate of real-world churn performance.

## Q50. Binary Iris Classification --- Virginica Probability and Threshold 0.6

``` python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()
X = iris.data
y = (iris.target == 2).astype(int)  # 1 = Virginica, 0 = not Virginica

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(X_train, y_train)

probabilities = model.predict_proba(X_test)[:, 1]
threshold = 0.6
predictions = (probabilities >= threshold).astype(int)

print("Virginica probabilities:", np.round(probabilities, 3))
print("Predictions at threshold 0.6:", predictions)
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions,
                            target_names=["Not Virginica", "Virginica"]))
```

## Q51. Simple Linear Regression --- Salary Dataset

Assumes `Salary.csv` has `years_of_experience` and `salary` columns.

``` python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Salary.csv")
X = df[["years_of_experience"]]
y = df["salary"]

model = LinearRegression()
model.fit(X, y)
pred = model.predict(X)
residuals = y - pred

print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])

plt.scatter(X.iloc[:, 0], y, label="Actual data")
plt.plot(X.iloc[:, 0], pred, label="Fitted line")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()

plt.scatter(pred, residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Salary")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.show()
```

## Q53. Multiple Linear Regression --- Car Prices (Age and Mileage)

Assumes `CarPrices.csv` contains `age`, `mileage`, and `price`.

``` python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("CarPrices.csv")
X = df[["age", "mileage"]]
y = df["price"]

model = LinearRegression()
model.fit(X, y)
pred = model.predict(X)
residuals = y - pred

print("Intercept:", model.intercept_)
print("Coefficients:", dict(zip(X.columns, model.coef_)))

plt.scatter(y, pred)
low = min(y.min(), pred.min())
high = max(y.max(), pred.max())
plt.plot([low, high], [low, high], "--")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.show()

plt.scatter(pred, residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Price")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.show()
```

## Q55. Linear Regression --- Car Prices, Evaluate MSE, R² and MAE

Assumes `CarPrices.csv` has `horsepower`, `age`, and `price`.

``` python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

df = pd.read_csv("CarPrices.csv")
X = df[["horsepower", "age"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, pred))
print("R²:", r2_score(y_test, pred))
print("MAE:", mean_absolute_error(y_test, pred))
```

## Q56. Linear Regression --- House Price with Location

Assumes `HouseData.csv` contains `num_rooms`, `square_feet`, `location`,
and `house_price`. Location is categorical, so encode it using one-hot
encoding.

``` python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

df = pd.read_csv("HouseData.csv")
X = df[["num_rooms", "square_feet", "location"]]
y = df["house_price"]

preprocess = ColumnTransformer([
    ("location", OneHotEncoder(handle_unknown="ignore"), ["location"])
], remainder="passthrough")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = make_pipeline(preprocess, LinearRegression())
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, pred))
print("R²:", r2_score(y_test, pred))
print("MAE:", mean_absolute_error(y_test, pred))
```

## Q57. Employee Salary Prediction --- Experience and Education Level

The PDF starts this question with `EmployeeData.csv`,
`years_of_experience`, `education_level`, and target `salary`, but the
remainder of the question is not visible in the extracted text. This
solution uses those stated fields to fit a linear regression model and
report MSE, R² and MAE.

``` python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

df = pd.read_csv("EmployeeData.csv")
X = df[["years_of_experience", "education_level"]]
y = df["salary"]

preprocess = ColumnTransformer([
    ("education", OneHotEncoder(handle_unknown="ignore"), ["education_level"])
], remainder="passthrough")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = make_pipeline(preprocess, LinearRegression())
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, pred))
print("R²:", r2_score(y_test, pred))
print("MAE:", mean_absolute_error(y_test, pred))
```

## Q62. PCA on 100 Random Samples with 3 Features

``` python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(42)
X = rng.normal(size=(100, 3))

X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Original shape:", X.shape)
print("Reduced shape:", X_pca.shape)
print("Explained variance ratio:", pca.explained_variance_ratio_)

plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA: 100 Samples, 3 Features Reduced to 2")
plt.show()
```

## Q63. PCA on Iris --- Print Explained Variance of All Components

``` python
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

iris = load_iris()
X = StandardScaler().fit_transform(iris.data)

pca = PCA()
X_pca = pca.fit_transform(X)

print("Explained variance ratio for each component:")
for i, ratio in enumerate(pca.explained_variance_ratio_, start=1):
    print(f"PC{i}: {ratio:.4f}")

print("Cumulative explained variance:",
      pca.explained_variance_ratio_.cumsum())

plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
         pca.explained_variance_ratio_, marker="o")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("PCA Explained Variance — Iris")
plt.xticks(range(1, len(pca.explained_variance_ratio_) + 1))
plt.grid(True)
plt.show()
```

## Q65. PCA on Iris --- 2D Plot

``` python
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = StandardScaler().fit_transform(iris.data)
X_pca = PCA(n_components=2).fit_transform(X)
y = iris.target

for i, name in enumerate(iris.target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=name)

plt.title("PCA-Reduced Iris Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()
```

## Q66. LDA on Wine Dataset --- Reduce to 2 Dimensions

``` python
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

wine = load_wine()
X = StandardScaler().fit_transform(wine.data)
y = wine.target

X_lda = LinearDiscriminantAnalysis(n_components=2).fit_transform(X, y)

for i, name in enumerate(wine.target_names):
    plt.scatter(X_lda[y == i, 0], X_lda[y == i, 1], label=name)

plt.title("LDA on Wine Dataset")
plt.xlabel("Linear Discriminant 1")
plt.ylabel("Linear Discriminant 2")
plt.legend()
plt.show()
```

## Q67. LDA on Breast Cancer Dataset --- Reduce to 1 Dimension

``` python
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

data = load_breast_cancer()
X = StandardScaler().fit_transform(data.data)
y = data.target

X_lda = LinearDiscriminantAnalysis(n_components=1).fit_transform(X, y)

plt.figure(figsize=(9, 4))
for cls, name in enumerate(data.target_names):
    values = X_lda[y == cls, 0]
    plt.scatter(values, [cls] * len(values), label=name, alpha=0.6)

plt.title("LDA on Breast Cancer Dataset (1 Component)")
plt.xlabel("Linear Discriminant 1")
plt.yticks([0, 1], data.target_names)
plt.legend()
plt.show()
```

## Q69. LDA on Iris --- Reduce to 2 Dimensions

``` python
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris = load_iris()
X = StandardScaler().fit_transform(iris.data)
y = iris.target

X_lda = LinearDiscriminantAnalysis(n_components=2).fit_transform(X, y)

for i, name in enumerate(iris.target_names):
    plt.scatter(X_lda[y == i, 0], X_lda[y == i, 1], label=name)

plt.title("LDA-Reduced Iris Dataset")
plt.xlabel("Linear Discriminant 1")
plt.ylabel("Linear Discriminant 2")
plt.legend()
plt.show()
```

## Q70. SVC Classification on Iris with Visualization

The SVC is trained on a train split. For a simple 2D visualization, the
plot uses the first two features (sepal length and width); model
training uses all four features.

``` python
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred, target_names=iris.target_names))

# Visualize test samples using their first two features.
# Colors indicate actual species; marker shapes indicate correct/incorrect predictions.
markers = ["o" if a == p else "x" for a, p in zip(y_test, pred)]
plt.figure(figsize=(8, 6))
for cls, name in enumerate(iris.target_names):
    mask = y_test == cls
    plt.scatter(X_test[mask, 0], X_test[mask, 1],
                label=f"Actual {name}", s=70)

for i, (a, p) in enumerate(zip(y_test, pred)):
    if a != p:
        plt.scatter(X_test[i, 0], X_test[i, 1],
                    marker="x", s=150, linewidths=2)

plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("SVC Iris Test Results (crosses = misclassified)")
plt.legend()
plt.show()
```

------------------------------------------------------------------------

## Important Notes

-   **CSV column names:** Verify the actual names in your files. In
    particular, diabetes target columns may be `label`, `Outcome`, or
    another name.
-   **Training vs testing:** Q1 explicitly requests training on the
    complete dataset. For other classification/regression questions, the
    examples use an 80/20 split where practical to estimate performance
    on unseen data.
-   **PCA vs LDA:** PCA is an unsupervised dimensionality-reduction
    method. LDA uses class labels and is supervised. For Iris, LDA can
    produce at most 2 components because there are 3 classes.
-   **Accuracy:** Results can vary with the dataset, split, and random
    seed. For the tiny churn dataset in Q48, the score is especially
    unstable.
