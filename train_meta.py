import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

def generate_task():
    n_features = np.random.randint(5, 15)
    n_informative = np.random.randint(2, n_features - 1)

    X, y = make_classification(
        n_samples=np.random.randint(200, 500),
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        class_sep=np.random.uniform(0.5, 2.0),
        random_state=np.random.randint(0, 1000)
    )
    return X, y


def get_meta_features(X, y):
    """
    Extract meta-features describing a dataset.
    """

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

   
    if not np.issubdtype(y.dtype, np.number):
        y = LabelEncoder().fit_transform(y)

    n_samples = X.shape[0]
    n_features = X.shape[1]


    mean_x = np.mean(X)
    std_x = np.std(X)
    var_x = np.var(X)

    min_x = np.min(X)
    max_x = np.max(X)

  
    missing_ratio = np.isnan(X).sum() / X.size

 
    try:
        corr = np.corrcoef(X, rowvar=False)

        corr = np.nan_to_num(corr)

        avg_corr = np.mean(np.abs(corr))

    except:

        avg_corr = 0

    # Target statistics
    classes = len(np.unique(y))

    class_counts = np.bincount(y)

    class_balance = np.max(class_counts) / np.sum(class_counts)

    meta = np.array([

        n_samples / 1000,

        n_features / 100,

        mean_x,

        std_x,

        var_x,

        min_x,

        max_x,

        missing_ratio,

        avg_corr,

        classes / 10,

        class_balance

    ], dtype=np.float32)

    return meta

def sample_config():
    model_type = np.random.choice(["rf", "knn", "nn"])

    if model_type == "rf":
        return {
            "type": "rf",
            "n_estimators": np.random.randint(50, 300),
            "max_depth": np.random.randint(3, 15)
        }

    elif model_type == "knn":
        return {
            "type": "knn",
            "n_neighbors": np.random.randint(1, 15)  # safer range
        }

    else:
        return {
            "type": "nn",
            "hidden_size": np.random.randint(20, 100),
            "lr": np.random.uniform(0.0005, 0.01)
        }

# -----------------------------
# Encoding
# -----------------------------
def config_to_vector(hp):
    vec = []

    if hp["type"] == "rf":
        vec += [1, 0, 0]
        vec += [hp["n_estimators"]/300, hp["max_depth"]/15, 0, 0]

    elif hp["type"] == "knn":
        vec += [0, 1, 0]
        vec += [0, 0, hp["n_neighbors"]/20, 0]

    else:
        vec += [0, 0, 1]
        vec += [0, 0, 0, hp["hidden_size"]/100]

    return np.array(vec)

# -----------------------------
# Evaluate models (FIXED)
# -----------------------------
def evaluate(X, y, hp):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    if hp["type"] == "rf":
        model = RandomForestClassifier(
            n_estimators=hp["n_estimators"],
            max_depth=hp["max_depth"]
        )

    elif hp["type"] == "knn":
        # 🔥 FIX: safe neighbors
        n_neighbors = min(hp["n_neighbors"], len(X_train))
        n_neighbors = max(1, n_neighbors)

        model = KNeighborsClassifier(n_neighbors=n_neighbors)

    else:
        model = MLPClassifier(
            hidden_layer_sizes=(hp["hidden_size"],),
            learning_rate_init=hp["lr"],
            max_iter=200
        )

    print("Evaluating:", hp)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return accuracy_score(y_test, preds)

def train_meta_model():
    meta_X, meta_y = [], []

    for _ in range(20):  # tasks
        X, y = generate_task()
        meta = get_meta_features(X, y)

        for _ in range(10):  # trials
            hp = sample_config()
            score = evaluate(X, y, hp)

            hp_vec = config_to_vector(hp)
            features = np.concatenate([meta, hp_vec])

            meta_X.append(features)
            meta_y.append(score)

    meta_X = np.array(meta_X, dtype=np.float32)
    meta_X = torch.from_numpy(meta_X)
    meta_y = np.array(meta_y, dtype=np.float32).reshape(-1,1)
    meta_y = torch.from_numpy(meta_y)
    print(meta_X.shape)
    meta = get_meta_features(X, y)
    print("Meta feature length:", len(meta))
    print("Meta features:", meta)

    model = nn.Sequential(
    nn.Linear(meta_X.shape[1], 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
    )

    optimizer = optim.Adam(model.parameters(),lr=0.001)
    loss_fn = nn.MSELoss()

    for _ in range(20):
        pred = model(meta_X)
        loss = loss_fn(pred, meta_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model

if __name__ == "__main__":

    print("Training Meta Model...")

    model = train_meta_model()

    torch.save(
        model.state_dict(),
        "surrogate.pt"
    )

    print("Model saved as surrogate.pt")