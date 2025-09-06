import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

def compute_item_similarity(R_train, use_normalize=True):
    item_vectors = R_train.T.toarray()
    item_norm = normalize(item_vectors, axis=1) if use_normalize else item_vectors
    return cosine_similarity(item_norm)

def truncated_svd_predict(R_train, n_components=20):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    U = svd.fit_transform(R_train)
    VT = svd.components_
    R_pred = np.dot(U, VT)
    return svd, R_pred
