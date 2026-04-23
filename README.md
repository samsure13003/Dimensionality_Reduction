# 🍷 Dimensionality Reduction on Wine Dataset

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)

A comparative study of three dimensionality reduction techniques — **PCA**, **Kernel PCA**, and **LDA** — applied to wine chemical data for customer segment classification. Each method reduces 13 features to 2 components, followed by Logistic Regression for classification.

---

## 📌 Project Overview

High-dimensional data poses challenges for both visualization and model training. This project explores how different dimensionality reduction strategies transform a 13-feature wine dataset into a 2D space, and compares their impact on downstream classification performance.

| Technique | Type | Supervised? | Key Idea |
|-----------|------|-------------|----------|
| PCA | Linear | ❌ No | Maximizes variance in projected space |
| Kernel PCA | Non-linear | ❌ No | Uses RBF kernel to capture non-linear structure |
| LDA | Linear | ✅ Yes | Maximizes class separability |

---

## 📂 Repository Structure

```
├── principal_component_analysis.ipynb   # PCA + Logistic Regression
├── kernel_pca.ipynb                     # Kernel PCA + Logistic Regression
├── linear_discriminant_analysis.ipynb   # LDA + Logistic Regression
├── Wine.csv                             # Dataset
└── README.md
```

---

## 📊 Dataset

**Wine.csv** — Chemical analysis of wines from 3 customer segments (classes 1, 2, 3).

| Property | Details |
|----------|---------|
| Samples | 178 |
| Features | 13 chemical attributes |
| Target | `Customer_Segment` (3 classes) |
| Class Distribution | Class 1: 59 · Class 2: 71 · Class 3: 48 |

**Features:** Alcohol, Malic Acid, Ash, Ash Alkalinity, Magnesium, Total Phenols, Flavanoids, Nonflavanoid Phenols, Proanthocyanins, Color Intensity, Hue, OD280, Proline

---

## ⚙️ Pipeline

All three notebooks follow the same pipeline:

```
Load Data → Train/Test Split (80/20) → Feature Scaling → 
Dimensionality Reduction (→ 2 components) → Logistic Regression → 
Confusion Matrix + Accuracy → Decision Boundary Visualization
```

---

## 🔬 Techniques

### 1. Principal Component Analysis (PCA)
`principal_component_analysis.ipynb`

PCA is an **unsupervised** linear technique that projects data onto the directions of maximum variance. It ignores class labels entirely, making it a general-purpose preprocessing step.

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test  = pca.transform(X_test)
```

### 2. Kernel PCA
`kernel_pca.ipynb`

Kernel PCA extends standard PCA to capture **non-linear** relationships by mapping data into a higher-dimensional space using the RBF (Radial Basis Function) kernel before applying PCA.

```python
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components=2, kernel='rbf')
X_train = kpca.fit_transform(X_train)
X_test  = kpca.transform(X_test)
```

### 3. Linear Discriminant Analysis (LDA)
`linear_discriminant_analysis.ipynb`

LDA is a **supervised** technique that explicitly uses class labels to find the projection that best separates the classes. It maximizes the between-class variance while minimizing within-class variance.

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2)
X_train = lda.fit_transform(X_train, y_train)
X_test  = lda.transform(X_test)
```

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **NumPy** — numerical operations
- **Pandas** — data loading and manipulation
- **Scikit-learn** — PCA, Kernel PCA, LDA, Logistic Regression, StandardScaler
- **Matplotlib** — decision boundary and scatter plot visualizations

---

## 🚀 Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/your-username/dimensionality-reduction-wine.git
cd dimensionality-reduction-wine
```

**2. Install dependencies**
```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

**3. Launch Jupyter**
```bash
jupyter notebook
```

**4. Run any notebook** — all three are self-contained and ready to execute top-to-bottom.

---

## 💡 Key Takeaways

- **LDA** tends to yield the best classification accuracy on this dataset because it leverages class label information during reduction, creating a more discriminative feature space.
- **PCA** provides a solid unsupervised baseline and works well when class information is unavailable.
- **Kernel PCA** is useful when data has non-linear structure, though its benefit over linear PCA depends on the dataset.
- Feature scaling (StandardScaler) is critical before applying any of these techniques.

---

## 👤 Author

**Samsur**
- GitHub: [@your-username](https://github.com/your-username)
- LinkedIn: [your-linkedin](https://linkedin.com/in/your-linkedin)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
