# Amanda Pollo Sarlo 22.121.047-9 
# Gianpietro Malheiros Braga 22.121.054-5 
# Lucca Kirsten da Costa 22.121.121-2 
# Marcella Rappoli Bedinelli 22.121.076-8


from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Carregar os dados do conjunto de dados Iris
data = load_iris()
features = data.data
target = data.target

# Treinamento do classificador sem PCA
Classificador = MLPClassifier(hidden_layer_sizes=(3), alpha=1, max_iter=2000)
Classificador.fit(features, target)

# Predições sem PCA
predicoes = Classificador.predict(features)

# Visualização sem PCA
plt.figure(figsize=(16, 8))
plt.subplot(2, 2, 1)
plt.scatter(features[:, 0], features[:, 1], c=target, marker='o', cmap='viridis')
plt.subplot(2, 2, 3)
plt.scatter(features[:, 0], features[:, 1], c=predicoes, marker='d', cmap='viridis', s=150)
plt.scatter(features[:, 0], features[:, 1], c=target, marker='o', cmap='viridis', s=15)

# Treinamento do classificador com PCA
pca = PCA(n_components=2, whiten=True, svd_solver='randomized')
pca_features = pca.fit_transform(features)
ClassificadorPCA = MLPClassifier(hidden_layer_sizes=(10), alpha=1, max_iter=1000)
ClassificadorPCA.fit(pca_features, target)

# Predições com PCA
predicoes_pca = ClassificadorPCA.predict(pca_features)

# Visualização com PCA
plt.subplot(2, 2, 2)
plt.scatter(pca_features[:, 0], pca_features[:, 1], c=target, marker='o', cmap='viridis')
plt.subplot(2, 2, 4)
plt.scatter(pca_features[:, 0], pca_features[:, 1], c=predicoes_pca, marker='d', cmap='viridis', s=150)
plt.scatter(pca_features[:, 0], pca_features[:, 1], c=target, marker='o', cmap='viridis', s=15)
plt.show()

# Matriz de confusão sem PCA
cm = confusion_matrix(target, predicoes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
disp.plot()
plt.title('Matriz de Confusão sem PCA')
plt.show()

# Matriz de confusão com PCA
cm_pca = confusion_matrix(target, predicoes_pca)
disp_pca = ConfusionMatrixDisplay(confusion_matrix=cm_pca, display_labels=data.target_names)
disp_pca.plot()
plt.title('Matriz de Confusão com PCA')
plt.show()
