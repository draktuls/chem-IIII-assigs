import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load data
gene_expression = pd.read_csv('./gene_expression.csv', header=None, delimiter=';', decimal=',')
labels = pd.read_csv('./label.csv', header=None)
labels = labels[0].map({1: "ALL", 2: "AML"})

y = labels
X = gene_expression.values

# Load gene names
with open('./geneNames.txt', 'r') as file:
    gene_names = [line.strip() for line in file]

# In case 'x' appears (invalid genes), replace
valid_indices = [i for i, name in enumerate(gene_names) if name != 'x']
X = X[:, valid_indices]
feature_names = [gene_names[i] for i in valid_indices]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=5555, stratify=y)

# Part 1 - Decision Tree

# Train Decision Tree
clf = DecisionTreeClassifier(random_state=5555, splitter="random")
clf.fit(X_train, y_train)

# Show the tree
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=feature_names, class_names=['AML', 'ALL'], filled=True, max_depth=3)
plt.title('Decision Tree (max depth=3 shown)')
plt.show(block=False)

print("\n\n--- Part 1 ---")

# Training accuracy
train_accuracy = clf.score(X, y)
print(f"Training Accuracy: {train_accuracy:.4f}")

# Most important gene
importance = clf.feature_importances_

# Top 5 most important genes
top_5_idx = np.argsort(importance)[-5:][::-1]  # Top 5 indices, descending
top_5_genes = [(feature_names[i], importance[i]) for i in top_5_idx]

print("\nTop 5 most important genes:")
for gene, score in top_5_genes:
    print(f"{gene}: {score:.4f}".replace('.', ','))


# Cross-validation accuracy (estimate real performance)
cross_val_scores = cross_val_score(clf, X, y, cv=5)
mean_cv_accuracy = np.mean(cross_val_scores)
print(f"Cross-Validation Accuracy: {mean_cv_accuracy:.4f}")

# Interpretation hint:
print("\nIs this gene really the one causing the cancer? Look up Golub et al., 1999.:")
print("""While CFD has been implicated in general cancer-related immune processes, it is not listed among the critical genes distinguishing AML and ALL in Golub et al., 1999.
Thus, while CFD may indirectly affect cancer progression, it is unlikely to be a causal gene in leukemia subtype classification in this context.""")

# Part 2 - PCA and reduced trees
print("\n\n--- Part 2 ---")

# Learn basis-matrix V using PCA
pca = PCA()
pca.fit(X)
V = pca.components_
print(f"\nPCA basis matrix V shape: {V.shape}")
print("First 3 PCA components (rows of V):")
print(V[:3])  # Show first 3 basis vectors

Ks = [1, 2, 5, 10, 20]
results = {}

for k in Ks:
    print(f"\n--- K = {k} ---")
    pca_k = PCA(n_components=k)
    Z = pca_k.fit_transform(X)

    X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(Z, y, test_size=0.3, random_state=42, stratify=y)

    clf_k = DecisionTreeClassifier(random_state=42)
    clf_k.fit(X_train_k, y_train_k)

    # Show the tree with correct feature names for PCs
    # pca_feature_names = [f"PC{i+1}" for i in range(k)]
    # plt.figure(figsize=(12,6))
    # plot_tree(clf_k, feature_names=pca_feature_names, filled=True, max_depth=3)
    # plt.title(f'Decision Tree for K={k} Components (max depth=3 shown)')
    # plt.show(block=False)

    # Training accuracy
    train_acc = clf_k.score(X_train_k, y_train_k)
    print(f"Training Accuracy with {k} components: {train_acc:.4f}".replace('.', ','))

    # Testing accuracy
    test_acc = clf_k.score(X_test_k, y_test_k)
    print(f"Testing Accuracy with {k} components: {test_acc:.4f}".replace('.', ','))

    # Store results
    results[k] = {
        'model': clf_k,
        'Z_train': X_train_k,
        'Z_test': X_test_k,
        'y_train': y_train_k,
        'y_test': y_test_k,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'pca_model': pca_k
    }

# Pick the best model based on simplicity and testing accuracy
best_k = max(results.keys(), key=lambda k: (results[k]['test_accuracy'], -k))
print(f"\nBest model chosen: K = {best_k}")

# Estimate real accuracy of the best model
clf_best = results[best_k]['model']
X_test_best = results[best_k]['Z_test']
y_test_best = results[best_k]['y_test']
test_accuracy_best = clf_best.score(X_test_best, y_test_best)
print(f"Testing Accuracy of Best Model (K={best_k}): {test_accuracy_best:.4f}".replace('.', ','))

# Part 3 - Extract active genes from discriminative components
print("\n\n--- Part 3: Extract Active Genes from Discriminative Components ---")

# Identify discriminative components (features used by the best tree)
used_features = np.unique(clf_best.tree_.feature)
used_features = used_features[used_features >= 0]  # remove -2 which indicates leaf nodes

# Map back to PCA components
pca_model_best = results[best_k]['pca_model']
components_used = [int(f) for f in used_features]

# For each discriminative component, cluster gene loadings
for idx in components_used:
    print(f"\nAnalyzing component PC{idx+1}...")
    component_vector = pca_model_best.components_[idx]
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(component_vector.reshape(-1, 1))

    # Identify active cluster (higher absolute values)
    cluster_centers = kmeans.cluster_centers_.flatten()
    active_cluster = np.argmax(np.abs(cluster_centers))

    active_genes_idx = np.where(clusters == active_cluster)[0]
    active_genes = [feature_names[i] for i in active_genes_idx]

    # Print active genes for GOrilla input
    print(f"Active genes for PC{idx+1} (to submit to GOrilla):")
    for gene in active_genes:
        print(gene)
    print("Total active genes:", len(active_genes))

input("Exit...")