import matplotlib.pyplot as plt
import numpy as np
import umap
from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tensorflow.keras.applications import EfficientNetV2B3
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

from modelbuilder import TransferLearningModelBuilder

height, width = 224, 224


def load_model():
    base_model = EfficientNetV2B3(
        weights="imagenet", include_top=False, input_shape=(height, width, 3)
    )
    feature_extractor = Model(
        inputs=base_model.input, outputs=base_model.layers[-1].output
    )
    return feature_extractor


def get_features(img_path, model, ewc=False):
    img = image.load_img(img_path, target_size=(height, width), color_mode="rgb")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = TransferLearningModelBuilder.preprocess_input(x, ewc=ewc)
    features = model.predict(x, verbose=0)
    return features.squeeze()


def scale_and_reduce(features_list, dim_reducer='tsne'):
    features_list_arr = np.array(features_list)
    num_samples = features_list_arr.shape[0]
    all_features_2d = features_list_arr.reshape(num_samples, -1)

    features_std = StandardScaler().fit_transform(all_features_2d)
    if dim_reducer == 'tsne':
        reducer = TSNE(n_components=2, random_state=0)
    elif dim_reducer == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=0)
    elif dim_reducer == 'pca':
        reducer = PCA(n_components=2, random_state=0)
    else:
        raise ValueError(f"Unknown dim_reducer: {dim_reducer}")
    return reducer.fit_transform(features_std)


def plot_features(low_dim_features, df):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("2D Visualization of <reducer> Features with Burst Information", fontsize=18)
    ax.set_xlabel("Feature 1", fontsize=14)
    ax.set_ylabel("Feature 2", fontsize=14)

    sc = ax.scatter(
        x=low_dim_features[:, 0],
        y=low_dim_features[:, 1],
        c=df["burst_type"].replace({"no_burst": 0}).astype(int),
        cmap="Spectral",
        s=10,
        alpha=0.7,
    )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Burst Type", fontsize=14)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.set_facecolor("#f4f4f4")
    plt.tight_layout()
    plt.show()


def train_and_plot_svm(low_dim_features, df):
    clf = SVC(kernel="rbf", C=1.0)
    clf.fit(low_dim_features, df["is_burst"])
    plot_decision_regions(low_dim_features, df["is_burst"].values, clf=clf, legend=2)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Reduced Dimensionality with Reducer")
    plt.show()
    return clf


def train_and_identify_misclassifications(clf, X, y, df):
    # Predicting on the test set
    y_pred = clf.predict(X)

    # Identifying misclassified indices
    misclassified_idx = np.where(y != y_pred)[0]

    # Returning wrongly classified images file paths
    return df.iloc[misclassified_idx]


def display_misclassified_images(df, num_images=9):
    # Sample random images from df
    df_sample = df.sample(n=num_images)
    misclassified_paths = df_sample["file_path"].values
    fig, axes = plt.subplots(
        nrows=int(np.sqrt(num_images)), ncols=int(np.sqrt(num_images)), figsize=(10, 10)
    )
    fig.suptitle("Misclassified Images", fontsize=14)

    for ax, (path, label) in zip(
        axes.ravel(), zip(misclassified_paths, df_sample["label"].values)
    ):
        img = image.load_img(path, target_size=(height, width))
        title = f"{path.split('/')[-1].split('.')[0]}"
        ax.set_title(title, fontsize=6)
        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

