import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

class ClusterFitter:

    @staticmethod
    def cluster_kmeans_predict():
        return 1

    @staticmethod
    def cluster_kmeans_fit():
        return 1

    @staticmethod
    def cluster_dbscan_predict(X_train, cluster_train, X_test, mdl_pca):
        """
        DBSCAN does not have something like stored cl
        :param X_train:ass centers. So we need to wrap it using a
        knn classifier
        :param cluster_train:
        :param X_test:
        :param mdl_cluster:
        :param mdl_pca: If not None, this will be used to PCA X_train and X_test
        :return:
        """
        reduced_data_test = X_test
        reduced_data_train = X_train

        if mdl_pca is not None:
            reduced_data_test = mdl_pca.transform(X_test)
            reduced_data_train = mdl_pca.transform(X_train)

        indicies_to_keep = np.array(cluster_train != -1)  # only keep indices with clear cluster (NOT -1)

        assert np.all(cluster_train, cluster_train[0]) != 1, "only one cluster in training data, this doesnt work"
        knn = KNeighborsClassifier(n_neighbors=5).fit(reduced_data_train[indicies_to_keep],
                                                      cluster_train[indicies_to_keep])
        cluster_test = knn.predict(reduced_data_test)
        return cluster_test

    @staticmethod
    def cluster_dbscan_fit(X, fixed_eps=0, pca_components=5, n_cluster_target=0,
                            enforce_labels=True):
        """

        :param X: The data holding features and samples
        :param fixed_eps: If you want to set a fixed distance for the DBSCAN algorithm use this. Set 0 if you want it
                          to be set automatically to match your number of clusters
        :param pca_components: PCA components used for clustering. If you run DBSCAN on a high number of dimensions it
        will run veeeeeeery slowly.
        :return: tuple of labels (=cluster id), the used distance and the classifier itself, model of the used pca
        """
        max_tries = 50
        used_eps = 0.5
        eps_delta = 0.1
        mdl_pca = PCA(n_components=pca_components).fit(X)
        reduced_data = mdl_pca.transform(X)
        for i in range(0, max_tries):

            if n_cluster_target is 0:
                #print("\t\t\tnumber of clusters unnkown, using eps = 0.5")
                used_eps = 0.5
            if fixed_eps is not 0:
                used_eps = fixed_eps

            mdl_cluster = DBSCAN(eps=used_eps, min_samples=100, metric="euclidean").fit(reduced_data)
            mdl_cluster.fit_predict(reduced_data)
            core_samples_mask = np.zeros_like(mdl_cluster.labels_, dtype=bool)
            core_samples_mask[mdl_cluster.core_sample_indices_] = True
            labels = mdl_cluster.labels_

            if n_cluster_target is 0:
                break

            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

            print("\t\t\ttrying DBSCAN with eps={} resulted into {} clusters, target was = {}".format(
                used_eps, n_clusters_, n_cluster_target
            ))

            if n_clusters_ is n_cluster_target or fixed_eps is not 0:
                break

            used_eps += eps_delta

        if enforce_labels is True:
            # predict one time to assign a label to the samples with -1
            labels = ClusterFitter.cluster_dbscan_predict(reduced_data, labels, reduced_data, mdl_pca=None)

        return labels, used_eps, mdl_cluster, mdl_pca
