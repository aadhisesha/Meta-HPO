#  MetaHPO: Meta-Learning Based Hyperparameter Optimization for Classification Models

MetaHPO is a **Meta-Learning Based Hyperparameter Optimization (Meta-HPO)** framework that applies knowledge learned from multiple synthetic classification tasks to recommend effective hyperparameter configurations for unseen datasets. The project demonstrates how **meta-learning** can improve the efficiency of hyperparameter optimization by learning transferable optimization patterns rather than starting the search from scratch for every dataset.

The framework generates a diverse collection of synthetic classification datasets using Scikit-learn, extracts dataset-level **meta-features**, evaluates different hyperparameter configurations across multiple machine learning models, and trains a **PyTorch neural surrogate model** to predict expected model performance. The trained surrogate model is then used to rank candidate hyperparameter configurations for new datasets before actual evaluation, enabling a more informed and efficient search process.

An interactive **Streamlit** application allows users to upload any classification dataset, automatically preprocess the data, compare supported machine learning models, identify the best-performing model, recommend optimized hyperparameters, and visualize optimization progress through interactive charts and analytics.

---

##  Key Features

-  Meta-learning based hyperparameter optimization using a neural surrogate model
-  Knowledge transfer from multiple synthetic classification tasks
- Automatic extraction of dataset meta-features
-  Hyperparameter optimization for multiple classification algorithms
-  Automatic preprocessing for uploaded datasets
-  Interactive Streamlit web application
-  Visualization of optimization progress and model performance
-  Offline surrogate model training with reusable learned knowledge

---

##  Supported Machine Learning Models

-  Random Forest Classifier
-  K-Nearest Neighbors (KNN)
-  Multi-Layer Perceptron (MLP)

---

##  Automatic Data Preprocessing

The application automatically performs:

- One-Hot Encoding for categorical features
- Missing value imputation
- Target label encoding
- Numeric feature conversion
- Dataset meta-feature extraction

This enables the framework to work with a wide range of real-world classification datasets.

---

##  Meta-Learning Workflow

```text
Synthetic Classification Tasks
            │
            ▼
Dataset Meta-Feature Extraction
            │
            ▼
Generate Random Hyperparameter Configurations
            │
            ▼
Evaluate Multiple ML Models
            │
            ▼
Train PyTorch Neural Surrogate Model
            │
            ▼
Save Trained Surrogate Model
────────────────────────────────────

User Uploads New Classification Dataset
            │
            ▼
Automatic Data Preprocessing
            │
            ▼
Extract Dataset Meta-Features
            │
            ▼
Generate Candidate Hyperparameters
            │
            ▼
Neural Surrogate Predicts Performance
            │
            ▼
Rank Candidate Configurations
            │
            ▼
Evaluate Top Configurations
            │
            ▼
Recommend Best Model & Hyperparameters
```

---

##  Project Structure

```
MetaHPO/
│
├── app.py                  # Streamlit application
├── train_meta.py           # Trains the neural surrogate model
├── surrogate.pt            # Saved surrogate model
├── requirements.txt        # Project dependencies
├── README.md
└── datasets/               # Sample datasets (optional)
```

---

##  Technology Stack

- Python
- PyTorch
- Scikit-learn
- Streamlit
- NumPy
- Pandas
- Matplotlib

---

##  Installation

Clone the repository

```bash
git clone https://github.com/yourusername/MetaHPO.git
cd MetaHPO
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

##  Training the Meta Model

Train the surrogate model using synthetic classification tasks.

```bash
python train_meta.py
```

This generates the trained surrogate model:

```
surrogate.pt
```

---

##  Running the Application

```bash
streamlit run app.py
```

---

##  Input Dataset Format

Upload any CSV dataset where:

- The last column represents the target class.
- All remaining columns are treated as input features.

Example:

| Age | Salary | Department | Purchased |
|-----|---------|------------|-----------|
|25|45000|Sales|Yes|
|42|70000|IT|No|

---

##  Application Output

The application provides:

- Best-performing classifier
- Recommended hyperparameters
- Classification accuracy
- Trial-wise optimization logs
- Optimization progress graphs
- Prediction vs Actual comparison

---

##  Project Objective

The objective of MetaHPO is to demonstrate how **meta-learning** can be used to improve hyperparameter optimization for classification problems. By learning from multiple synthetic tasks, the framework transfers optimization knowledge to unseen datasets, enabling more efficient hyperparameter selection and model comparison.

---

##  Future Enhancements

- Support for regression tasks
- Additional machine learning algorithms (SVM, XGBoost, LightGBM, CatBoost)
- Bayesian Optimization integration
- Gradient-based meta-learning (MAML)
- Multi-objective hyperparameter optimization
- Parallel hyperparameter evaluation
- Persistent meta-dataset storage
- Cross-validation based optimization

---

##  Author

**Aadhisesha D**

B.E. Computer Science Engineering  
College of Engineering Guindy (Anna University)

---

##  License

This project is developed for educational, research, and learning purposes.
