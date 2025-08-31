# Q: What is your name
A: My name is Mascot.

# Q: Who built you
A: I was built by Sidharth.

# Python Programming (Intermediate/Advanced)


**Data Structures & Comprehensions:** Python's built-in data types
(lists, dictionaries, sets, tuples) can be manipulated concisely using
comprehensions. For example, `[x**2 for x in range(10)]` builds a list
of squares. List/dict/set comprehensions allow filtering and mapping in
a single expression. They yield readable, efficient code for creating
collections.

**Generators & Iterators:** Generators are functions using `yield` to
produce a lazy iterator. They yield values one at a time without storing
all results in memory. For instance:

``` python
def count_up_to(n):
    i = 0
    while i <= n:
        yield i
        i += 1
```

Each call to `next()` on this generator resumes execution after the last
`yield`. Generators are ideal for streaming large datasets (e.g. reading
lines from a file) without exhausting memory. When the `yield` statement
is hit, the function suspends execution and returns the yielded value to
the caller, unlike `return` which ends the function. This allows
resuming where it left off on subsequent calls.

**Higher-Order Functions & Decorators:** Python treats functions as
first-class objects, enabling use of higher-order functions (functions
taking other functions as arguments). Decorators wrap a function to
modify its behavior without changing its code. For example, a logging
decorator might record inputs/outputs transparently. Decorators follow
the form:

``` python
def decorator(func):
    def wrapper(*args, **kwargs):
        # pre-processing
        result = func(*args, **kwargs)
        # post-processing
        return result
    return wrapper

@decorator
def my_function(...):
    ...
```

This wraps `my_function` with extra behavior. The order of multiple
decorators matters (innermost first). Decorators improve modularity
(e.g. for logging, timing, caching) and *"allow modifying or extending
the behavior of functions/methods without changing their code"*.

**Object-Oriented Features:** Python's classes support inheritance,
polymorphism, and special ("dunder") methods. Advanced patterns
(factory, singleton, etc.) can be implemented via class or metaclass
decorators. Use of `@staticmethod`, `@classmethod`, and property
decorators (`@property`) helps encapsulate functionality. Special
methods like `__repr__`, `__eq__`, and operator overloading (`__add__`)
can make classes more Pythonic.

**Context Managers:** Use `with` statements and context managers
(implementing `__enter__` and `__exit__`) for resource management
(files, locks, database sessions). For example:

``` python
with open('data.csv') as f:
    data = f.read()
```

ensures the file is closed automatically, even if errors occur.

**Concurrency (Threads/Async):** For I/O-bound tasks, Python's `asyncio`
library enables asynchronous programming (`async def`, `await` syntax)
for handling many network or file operations efficiently. For CPU-bound
work, `concurrent.futures.ThreadPoolExecutor` or `ProcessPoolExecutor`
can run tasks in parallel threads or processes (note Python's GIL limits
true parallelism in threads for CPU).

**Type Hints & Static Typing:** Modern Python supports optional type
hints (PEP 484) to improve code clarity and tooling. Best practices
include using abstract types (e.g. `Iterable[int]`, `Mapping[str, Any]`)
for parameters and concrete built-ins for returns. For instance:

``` python
from typing import Iterable

def process_data(values: Iterable[int]) -> list[int]:
    return [v*2 for v in values]
```

Use the `|` syntax for unions (Python 3.10+), e.g. `str | None` instead
of `Optional[str]`. Prefer built-in generics (`list[int]`,
`dict[str, int]`) over those from `typing` for brevity.

**Best Practices & Tools:** Follow PEP 8 coding standards (use linters
like `flake8` or `pylint`). Write docstrings and use type annotations to
improve readability and catch errors early. Use version control
(e.g. Git), virtual environments (`venv`/`conda`) to manage
dependencies, and automated testing (`pytest` or `unittest`).

## Data Handling with Pandas and NumPy

**NumPy:** The foundation of scientific Python is NumPy, providing
efficient `ndarray` for numerical data and operations (linear algebra,
broadcasting). Use NumPy for vectorized computations (avoiding Python
loops) for speed. Example: `arr = np.array([1,2,3]); arr.mean()`
computes the average efficiently.

**Pandas DataFrame:** Pandas' `DataFrame` is the core tabular data
structure. You can load data from files/SQL with methods like
`pd.read_csv()`. For example:

``` python
import pandas as pd
df = pd.read_csv('data.csv')   # loads CSV into a DataFrame
```

Common operations include indexing/filtering rows (`df[df['col'] > 0]`),
selecting columns (`df[['col1','col2']]`), aggregation
(`df.groupby('category').sum()`), and merging/joining tables
(`pd.merge(df1, df2, on='key')`). Pivot tables and reshaping
(`pivot_table`, `stack`, `melt`) help reorganize data. Pandas integrates
with NumPy for mathematical ops and with plotting libraries
(Matplotlib/Seaborn) for visualization.

**Data Cleaning:** Pandas provides methods for handling missing data
(`df.dropna()`, `df.fillna()`, or interpolation) and duplicates
(`df.drop_duplicates()`). Converting datatypes
(e.g. `df['date'] = pd.to_datetime(df['date']`) is straightforward. For
large data, use chunked reading (`read_csv` with `chunksize`) or
out-of-core solutions (Dask, Vaex).

# Machine Learning Concepts

## Data Preprocessing

Before modeling, preprocess data carefully to avoid common pitfalls.
Split data into training and test sets *before* applying transformations
to prevent data leakage. For example, fit scalers or encoders on
training data only, then apply to test data. Tools like **scikit-learn
Pipelines** enforce this: they chain transformations and estimator into
one object. Example pipeline:

``` python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = make_pipeline(StandardScaler(), LogisticRegression())
pipeline.fit(X_train, y_train)
score = pipeline.score(X_test, y_test)
```

This automatically applies the scaler to train and test data
consistently. Pipelines make code concise and prevent forgetting steps.
They also support joint hyperparameter search across all steps.

Key preprocessing steps include: - **Handling Missing Values:** Impute
missing features instead of dropping data when possible. Scikit-learn's
`SimpleImputer` can fill missing values with strategies like mean/median
or a constant. For example: `SimpleImputer(strategy="median")` computes
medians on training data and imputes both training and test sets
consistently.\
- **Encoding Categorical Data:** Convert categorical features to
numerical form. Common techniques include one-hot encoding
(`OneHotEncoder`) and ordinal encoding (`OrdinalEncoder`). One-hot is
useful when no ordinal relationship; label/ordinal for tree-based models
or when categories are naturally ordered. Be careful with
high-cardinality features (group infrequent categories) to avoid
excessive dimensions.\
- **Feature Scaling:** Standardize numeric features to have zero mean
and unit variance (`StandardScaler`) or to a fixed range
(`MinMaxScaler`) if the model (e.g. SVM, neural nets) is sensitive to
feature scales. Always fit scalers on training data only.\
- **Feature Engineering/Selection:** Generate new features (polynomial
features, interaction terms with `PolynomialFeatures`) or reduce
dimensionality (e.g. PCA, t-SNE). Use domain knowledge to create
meaningful features. Avoid selecting features based on the entire
dataset -- selection should be done inside cross-validation or pipeline
to avoid leakage.

## Modeling: Supervised Learning

**Linear Models:** Simple yet powerful.\
- *Linear Regression* for continuous targets: models `y ≈ w·x + b`. Use
ordinary least squares or regularized versions (Ridge/Lasso) to prevent
overfitting.\
- *Logistic Regression* for binary classification: applies a logistic
sigmoid to linear combination of features. It can incorporate L1/L2
regularization.\
- *Support Vector Machines (SVMs):* Effective for classification (and
regression) with kernels (linear, polynomial, RBF). They find decision
boundaries maximizing margin. SVMs often require feature scaling and can
be slower on very large datasets.

**Tree-Based Methods:**\
- *Decision Trees:* Recursive partitioning of features. Easy to
interpret but prone to overfitting if unpruned. Control complexity via
depth or number of leaf nodes.\
- *Ensembles:* Combine many trees for robust performance. Random Forests
(bagged trees) and Gradient Boosting Machines (e.g. XGBoost, LightGBM)
are popular for structured data. They automatically handle mixed types
and non-linearities, and often perform well with minimal feature
engineering.

**Nearest Neighbors:** K-Nearest Neighbors (KNN) classifies or regresses
based on closest training points. Simple, but scales poorly with data
size and sensitive to distance metrics and feature scaling.

**Neural Networks (MLPs):** Multi-layer perceptrons (feedforward neural
networks) can model complex functions. Use frameworks (TensorFlow/Keras,
PyTorch) to build and train. Even a simple architecture:

``` python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, validation_split=0.1)
```

In PyTorch, one defines a model by subclassing `nn.Module` and
implementing `forward()`. For example:

``` python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28*28, 512), nn.ReLU(),
            nn.Linear(512, 10)
        )
    def forward(self, x):
        x = x.view(-1, 28*28)
        return self.net(x)

model = MyModel().to(device)
```

## Modeling: Unsupervised Learning

-   **Clustering:** E.g. *k-means* partitions data into k clusters by
    minimizing within-cluster variance. Other algorithms include DBSCAN
    (density-based) and Gaussian Mixture Models. Evaluate clustering
    with metrics like silhouette score.\
-   **Dimensionality Reduction:** *Principal Component Analysis (PCA)*
    finds orthogonal axes maximizing variance, useful for compression or
    visualization. *t-SNE* and *UMAP* project high-dimensional data to
    2D/3D for visualization of clusters.\
-   **Others:** Association rule learning, anomaly detection (Isolation
    Forest, One-Class SVM), etc., used in specific contexts (e.g. fraud
    detection).

## Model Evaluation & Validation

**Train/Test Split & Cross-Validation:** Always set aside unseen data
for final testing. K-fold cross-validation (CV) is used to estimate
generalization; it repeatedly splits data into train/validation folds.
Use tools like `cross_val_score` or `GridSearchCV` in scikit-learn.
Never use test data to tune parameters -- this causes *data leakage*.
Instead, use CV on training data and keep test data untouched.

**Metrics:** Choose metrics aligned with your task and data balance.
Common metrics include:

-   *Classification:* **Accuracy** measures fraction of correct
    predictions, but can be misleading on imbalanced classes.
    **Precision** and **Recall** quantify performance on the positive
    class. The *F1 score* is the harmonic mean of precision and recall.
    For probabilistic classifiers, the ROC curve and AUC measure
    trade-off across thresholds.
-   *Regression:* **Mean Absolute Error (MAE)** and **Mean Squared Error
    (MSE)/Root MSE** measure prediction errors. MSE penalizes larger
    errors more heavily. **R²** (coefficient of determination) indicates
    the fraction of variance explained.\
-   *Clustering:* Metrics like *silhouette score* or *Davies-Bouldin
    index* assess cluster cohesion and separation.

**Best Practices:**\
- **Avoid Data Leakage:** Do not preprocess on the entire dataset before
splitting. Apply transformations (scaling, selection) within the
training fold only.\
- **Use Pipelines:** As mentioned, scikit-learn `Pipeline` enforces
correct sequence of transforms and estimation and avoids forgetting to
apply the same transforms to test data.\
- **Hyperparameter Tuning:** Use grid or random search
(e.g. `GridSearchCV`), incorporating CV. Tune hyperparameters
(regularization strength, number of trees, learning rate) based on
validation performance, not on test data.\
- **Reproducibility:** Set random seeds and document package versions.
Logging experiments (e.g. TensorBoard, MLflow) helps track experiments.

# Key Libraries and Frameworks

-   **Scikit-learn:** A comprehensive Python library for traditional ML,
    including preprocessing, model algorithms, model selection, and
    evaluation tools. It offers a unified API (`.fit()/.predict()`)
    across models.\
-   **TensorFlow / Keras:** A deep learning framework with high-level
    (Keras) and low-level APIs. Keras (now integrated into TF) makes
    model building intuitive with layers and Sequential/Functional APIs.
    Key workflow: define model architecture, call
    `compile(optimizer, loss, metrics)`, then `fit` on training data.
    TensorFlow supports GPU/TPU acceleration, advanced models (CNNs,
    RNNs, Transformers), and deployment (SavedModel, TF Lite).\
-   **PyTorch:** A deep learning library known for its dynamic
    computation graph and Pythonic style. Models subclass `nn.Module`;
    use `autograd` for differentiation.\
-   **Other Tools:** Pandas, NumPy, Matplotlib, Seaborn, XGBoost,
    LightGBM, Hugging Face Transformers.

# Application Areas

-   **Computer Vision:** Image classification, object detection, and
    segmentation (e.g., autonomous driving, medical image analysis)
    using CNNs.\
-   **Natural Language Processing (NLP):** Text classification, language
    translation, and speech recognition.\
-   **Recommender Systems:** Netflix, Amazon, Spotify personalization.\
-   **Healthcare:** Medical diagnosis, imaging, and drug discovery.\
-   **Finance:** Fraud detection and algorithmic trading.\
-   **Robotics and Autonomous Systems:** Drones, robotics, smart
    manufacturing.\
-   **Time Series & Forecasting:** Stock prices, weather prediction, and
    demand forecasting.
