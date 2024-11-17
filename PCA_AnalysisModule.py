#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 21:13:04 2024

@author: Arie Pyasik, APA.inc

This Module calculates the PCA (Principle Component Analysis) of the spectral 
data fo dimentionality reduction and then clusters it for classification.
"""

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def train_svm_classifier(pca_data, labels, use_grid_search=False):
    """
    Trains an SVM classifier on PCA-transformed data with options for hyperparameter tuning.
    
    Parameters:
    - pca_data (array-like): The PCA-transformed data (features).
    - labels (array-like): Target labels for classification.
    - use_grid_search (bool): Whether to perform grid search for hyperparameter tuning (default: False).
    
    Returns:
    - classifier: Trained SVM classifier.
    - metrics: Dictionary containing 'accuracy' and 'classification_report'.
    """
    
    # Step 1: Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(pca_data, labels, test_size=0.3, random_state=42)

    # Step 2: Initialize SVM Classifier
    svm_classifier = SVC(kernel='rbf', gamma='scale', C=1.0, random_state=42)

    # Step 3: Grid Search for Hyperparameter Tuning (if specified)
    if use_grid_search:
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 0.1, 0.01, 0.001],
            'kernel': ['rbf']
        }
        grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', verbose=1)
        grid_search.fit(X_train, y_train)
        svm_classifier = grid_search.best_estimator_
        print("Best Parameters:", grid_search.best_params_)
    
    # Step 4: Train the classifier
    svm_classifier.fit(X_train, y_train)

    # Step 5: Predict and Evaluate
    y_pred = svm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Print metrics
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", report)

    # Package metrics into a dictionary for easy access
    metrics = {
        'accuracy': accuracy,
        'classification_report': report
    }

    return svm_classifier, metrics


def pca_kmeans_clustering(data, n_components=3, n_clusters=3):
    """
    Perform PCA on the data, reducing it to `n_components` dimensions,
    then apply K-means clustering with `n_clusters` clusters.

    Parameters:
    - data (array-like): Input data to be clustered. Should be a 2D array where rows are samples and columns are features.
    - n_components (int): Number of dimensions to reduce the data to with PCA.
    - n_clusters (int): Number of clusters to form with K-means.

    Returns:
    - clustered_data (array-like): The transformed data after PCA.
    - labels (array-like): Array of cluster labels for each sample in the input data.
    """

    # Step 1: Perform PCA
    pca = PCA(n_components=n_components)
    cols_without_nan = np.unique(np.argwhere((np.isnan(data) == False))[:, 0])
    data = data[cols_without_nan, :] #remove nan
    transformed_data = pca.fit_transform(data)

    # Step 2: Perform K-means clustering on the reduced data
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(transformed_data)

    return cols_without_nan, transformed_data, labels



def plot_svm_decision_boundaries_3d(classifier, pca_data, labels):
    """
    Plots the 3D decision boundaries of a trained SVM classifier using the first three
    principal components of the data.
    
    Parameters:
    - classifier: Trained SVM classifier.
    - pca_data (array-like): The PCA-transformed data (features), assuming exactly 3 components.
    - labels (array-like): Target labels for classification.
    """
    
    # Set up the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot data points with their labels
    scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], c=labels, cmap=plt.cm.coolwarm, edgecolor='k', s=40)
    ax.legend(*scatter.legend_elements(), title="Classes")

    # Set labels
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title('SVM Decision Boundaries with First 3 Principal Components')

    # Create a mesh grid to plot decision boundaries in 3D
    x_min, x_max = pca_data[:, 0].min() - 1, pca_data[:, 0].max() + 1
    y_min, y_max = pca_data[:, 1].min() - 1, pca_data[:, 1].max() + 1
    z_min, z_max = pca_data[:, 2].min() - 1, pca_data[:, 2].max() + 1
    xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, 0.5),
                             np.arange(y_min, y_max, 0.5),
                             np.arange(z_min, z_max, 0.5))

    # Predict the class of each point in the grid
    grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    Z = classifier.predict(grid)
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    ax.contourf(xx[:, :, 0], yy[:, :, 0], Z[:, :, int(Z.shape[2] / 2)], alpha=0.3, cmap=plt.cm.coolwarm)

    plt.show()