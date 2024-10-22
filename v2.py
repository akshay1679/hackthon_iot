import numpy as np
import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import random

# ---------------------------
# Step 1: Whale Optimization Algorithm (WOA)
# ---------------------------
def whale_optimization_algorithm(obj_func, dim, n_whales=10, max_iter=30):  # Increased iterations
    lb, ub = -1, 1
    whales = np.random.uniform(lb, ub, (n_whales, dim))
    leader = whales[np.argmin([obj_func(x) for x in whales])]

    for t in range(max_iter):
        a = 2 - t * (2 / max_iter)
        for i in range(n_whales):
            r = random.random()
            A = 2 * a * r - a
            C = 2 * r
            p = random.random()
            if p < 0.5:
                D = np.abs(C * leader - whales[i])
                whales[i] = leader - A * D
            else:
                whales[i] = np.random.uniform(lb, ub, dim)

        fitness = [obj_func(x) for x in whales]
        leader = whales[np.argmin(fitness)]

    return leader

# ---------------------------
# Step 2: Grey Wolf Optimization (GWO)
# ---------------------------
def grey_wolf_optimization(obj_func, dim, n_wolves=10, max_iter=30):  # Increased iterations
    lb, ub = -1, 1
    wolves = np.random.uniform(lb, ub, (n_wolves, dim))
    alpha, beta, delta = wolves[:3]

    for t in range(max_iter):
        a = 2 - t * (2 / max_iter)
        for i in range(n_wolves):
            A1, A2, A3 = 2 * a * random.random() - a, 2 * a * random.random() - a, 2 * a * random.random() - a
            C1, C2, C3 = 2 * random.random(), 2 * random.random(), 2 * random.random()

            D_alpha = np.abs(C1 * alpha - wolves[i])
            D_beta = np.abs(C2 * beta - wolves[i])
            D_delta = np.abs(C3 * delta - wolves[i])

            wolves[i] = (alpha - A1 * D_alpha + beta - A2 * D_beta + delta - A3 * D_delta) / 3

        fitness = [obj_func(x) for x in wolves]
        sorted_indices = np.argsort(fitness)
        alpha, beta, delta = wolves[sorted_indices[:3]]

    return alpha

# ---------------------------
# Step 3: Fitness Function with Random Forest
# ---------------------------
def fitness_func(solution, X, y):
    selected_features = [i for i, bit in enumerate(solution) if bit > 0]
    if len(selected_features) == 0:
        return float('inf')  # Invalid solution

    X_selected = X[:, selected_features]
    model = RandomForestClassifier(n_estimators=10)  # Use Random Forest for quick evaluation
    model.fit(X_selected, y)
    return 1 - model.score(X_selected, y)  # Minimize error

# ---------------------------
# Step 4: Feature Selection Optimization
# ---------------------------
def optimize_feature_selection(X, y):
    dim = X.shape[1]
    best_solution_woa = whale_optimization_algorithm(lambda x: fitness_func(x, X, y), dim)
    best_solution_gwo = grey_wolf_optimization(lambda x: fitness_func(x, X, y), dim)
    return np.where((best_solution_woa + best_solution_gwo) > 0, 1, 0)

# ---------------------------
# Step 5: Main Program
# ---------------------------
if __name__ == '__main__':
    # Simulated IoT dataset (features and labels)
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Feature Selection using Hybrid Optimization
    selected_features = optimize_feature_selection(X_train, y_train)
    X_train_opt = X_train[:, selected_features == 1]
    X_test_opt = X_test[:, selected_features == 1]

    # Build and Train a Deeper Neural Network
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, input_dim=X_train_opt.shape[1], activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_opt, y_train, epochs=20, validation_data=(X_test_opt, y_test), verbose=1)  # Increased epochs

    # Evaluate the Model
    y_pred = (model.predict(X_test_opt) > 0.5).astype("int32")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy: ", accuracy_score(y_test, y_pred))
