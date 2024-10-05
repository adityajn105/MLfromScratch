<div align="center">

<h1>🧠 MLfromScratch</h1>

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Library-green?style=flat-square&logo=numpy&logoColor=white)
![License](https://img.shields.io/github/license/adityajn105/MLfromScratch?style=flat-square)
![GitHub contributors](https://img.shields.io/github/contributors/adityajn105/MLfromScratch?style=flat-square)

</div>

**MLfromScratch** is a library designed to help you learn and understand machine learning algorithms by building them from scratch using only `NumPy`! No black-box libraries, no hidden magic—just pure Python and math. It's perfect for beginners who want to see what's happening behind the scenes of popular machine learning models.

🔗 **[Explore the Documentation](https://github.com/adityajn105/MLfromScratch/wiki)**

---

## 📦 Package Structure

Our package structure is designed to look like `scikit-learn`, so if you're familiar with that, you'll feel right at home!

### 🔧 Modules and Algorithms (Explained for Beginners) <br></br>
#### 📈 **1. Linear Models (`linear_model`)**

- **LinearRegression** ![Linear Regression](https://img.shields.io/badge/Linear%20Regression-blue?style=flat-square&logo=mathworks): Imagine drawing a straight line through a set of points to predict future values. Linear Regression helps in predicting something like house prices based on size.
  
- **SGDRegressor** ![SGD](https://img.shields.io/badge/SGD-Fast-blue?style=flat-square&logo=rocket): A fast way to do Linear Regression using Stochastic Gradient Descent. Perfect for large datasets.

- **SGDClassifier** ![Classifier](https://img.shields.io/badge/SGD-Classifier-yellow?style=flat-square&logo=target): A classification algorithm predicting categories like "spam" or "not spam." <br></br>


#### 🌳 **2. Decision Trees (`tree`)**

- **DecisionTreeClassifier** ![Tree](https://img.shields.io/badge/Tree-Classifier-brightgreen?style=flat-square&logo=leaf): Think of this as playing 20 questions to guess something. A decision tree asks yes/no questions to classify data.
  
- **DecisionTreeRegressor** ![Regressor](https://img.shields.io/badge/Tree-Regressor-yellowgreen?style=flat-square&logo=mathworks): Predicts a continuous number (like temperature tomorrow) based on input features. <br></br>


#### 👥 **3. K-Nearest Neighbors (`neighbors`)**

- **KNeighborsClassifier** ![KNN](https://img.shields.io/badge/KNN-Classifier-9cf?style=flat-square&logo=people-arrows): Classifies data by looking at the 'k' nearest neighbors to the new point.

- **KNeighborsRegressor** ![KNN](https://img.shields.io/badge/KNN-Regressor-lightblue?style=flat-square&logo=chart-bar): Instead of classifying, it predicts a number based on nearby data points. <br></br>


#### 🧮 **4. Naive Bayes (`naive_bayes`)**

- **GaussianNB** ![Gaussian](https://img.shields.io/badge/GaussianNB-fast-brightgreen?style=flat-square&logo=matrix): Works great for data that follows a normal distribution (bell-shaped curve).

- **MultinomialNB** ![Multinomial](https://img.shields.io/badge/MultinomialNB-text-ff69b4?style=flat-square&logo=alphabetical-order): Ideal for text classification tasks like spam detection. <br></br>


#### 📊 **5. Clustering (`cluster`)**

- **KMeans** ![KMeans](https://img.shields.io/badge/KMeans-Clustering-ff69b4?style=flat-square&logo=group): Groups data into 'k' clusters based on similarity.
  
- **AgglomerativeClustering** ![Agglomerative](https://img.shields.io/badge/Agglomerative-Hierarchical-blueviolet?style=flat-square&logo=chart-bar): Clusters by merging similar points until a single large cluster is formed.

- **DBSCAN** ![DBSCAN](https://img.shields.io/badge/DBSCAN-Noise%20Filtering-blue?style=flat-square&logo=waves): Groups points close to each other and filters out noise. No need to specify the number of clusters!

- **MeanShift** ![MeanShift](https://img.shields.io/badge/MeanShift-Clustering-yellowgreen?style=flat-square&logo=sort-amount-up): Shifts data points toward areas of high density to find clusters. <br></br>


#### 🌲 **6. Ensemble Methods (`ensemble`)**

- **RandomForestClassifier** ![RandomForest](https://img.shields.io/badge/Random%20Forest-Classifier-brightgreen?style=flat-square&logo=forest): Combines multiple decision trees to make stronger decisions.
  
- **RandomForestRegressor** ![RandomForest](https://img.shields.io/badge/Random%20Forest-Regressor-lightblue?style=flat-square&logo=tree): Predicts continuous values using an ensemble of decision trees.

- **GradientBoostingClassifier** ![GradientBoosting](https://img.shields.io/badge/Gradient%20Boosting-Classifier-9cf?style=flat-square&logo=chart-line): Builds trees sequentially, each correcting errors made by the last.

- **VotingClassifier** ![Voting](https://img.shields.io/badge/Voting-Classifier-orange?style=flat-square&logo=thumbs-up): Combines the results of multiple models to make a final prediction. <br></br>


#### 📐 **7. Metrics (`metrics`)**

Measure your model’s performance:

- **accuracy_score** ![Accuracy](https://img.shields.io/badge/Accuracy-High-brightgreen?style=flat-square&logo=bar-chart): Measures how many predictions your model got right.

- **f1_score** ![F1 Score](https://img.shields.io/badge/F1_Score-Balance-lightgreen?style=flat-square&logo=scales): Balances precision and recall into a single score.

- **roc_curve** ![ROC](https://img.shields.io/badge/ROC-Curve-orange?style=flat-square&logo=wave): Shows the trade-off between true positives and false positives. <br></br>


#### ⚙️ **8. Model Selection (`model_selection`)**

- **train_test_split** ![TrainTestSplit](https://img.shields.io/badge/Train_Test_Split-blueviolet?style=flat-square&logo=arrows): Splits your data into training and test sets.

- **KFold** ![KFold](https://img.shields.io/badge/KFold-CrossValidation-blue?style=flat-square&logo=matrix): Trains the model in 'k' iterations for better validation. <br></br>


#### 🔍 **9. Preprocessing (`preprocessing`)**

- **StandardScaler** ![StandardScaler](https://img.shields.io/badge/StandardScaler-Normalization-ff69b4?style=flat-square&logo=arrows-v): Standardizes your data so it has a mean of 0 and a standard deviation of 1.

- **LabelEncoder** ![LabelEncoder](https://img.shields.io/badge/LabelEncoder-Classification-yellow?style=flat-square&logo=code): Converts text labels into numerical labels (e.g., "cat", "dog").<br></br>


#### 🧩 **10. Dimensionality Reduction (`decomposition`)**

Dimensionality Reduction helps in simplifying data while retaining most of its valuable information. By reducing the number of features (dimensions) in a dataset, it makes data easier to visualize and speeds up machine learning algorithms.

- **PCA (Principal Component Analysis)** ![PCA](https://img.shields.io/badge/PCA-PrincipalComponentAnalysis-orange?style=flat-square&logo=chart-line): PCA reduces the number of dimensions by finding new uncorrelated variables called principal components. It projects your data onto a lower-dimensional space while retaining as much variance as possible. <br></br>
  - **How It Works**: PCA finds the axes (principal components) that maximize the variance in your data. The first principal component captures the most variance, and each subsequent component captures progressively less.
  - **Use Case**: Use PCA when you have many features, and you want to simplify your dataset for better visualization or faster computation. It is particularly useful when features are highly correlated.

---

## 🎯 Why Use This Library?

- **Learning-First Approach**: If you're a beginner and want to *understand* machine learning, this is the library for you. No hidden complexity, just code.
- **No Hidden Magic**: Everything is written from scratch, so you can see exactly how each algorithm works.
- **Lightweight**: Uses only `NumPy`, making it fast and easy to run. <br></br>

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/adityajn105/MLfromScratch.git

# Navigate to the project directory
cd MLfromScratch

# Install the required dependencies
pip install -r requirements.txt
```
<br></br>

## 👨‍💻 Author
This project is maintained by [Aditya Jain](https://adityajain.me/)<br></br>

## 🧑‍💻 Contributors
Constributor: [Subrahmanya Gaonkar](https://github.com/negativenagesh)

We welcome contributions from everyone, especially beginners! If you're new to open-source, don’t worry—feel free to ask questions, open issues, or submit a pull request. <br></br>

## 🤝 How to Contribute
1. Fork the repository.
2. Create a new branch (git checkout -b feature-branch).
3. Make your changes and commit (git commit -m "Added new feature").
4. Push the changes (git push origin feature-branch).
5. Submit a pull request and explain your changes. <br></br>

## 📄 License
This project is licensed under the [MIT License](https://github.com/adityajn105/MLfromScratch/blob/master/LICENSE) - see the LICENSE file for details.