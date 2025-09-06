from src.data_utils import create_synthetic_ratings, preprocess_ratings
from src.models import truncated_svd_predict
import pickle, os
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

print("Loading synthetic dataset...")
ratings = create_synthetic_ratings()
ratings_proc, user2idx, item2idx, R = preprocess_ratings(ratings)

print("Splitting train and test...")
train_df, test_df = train_test_split(ratings_proc, test_size=0.2, random_state=42)
R_train = csr_matrix((train_df['rating'], (train_df['u_idx'], train_df['i_idx'])), shape=R.shape)

print("Training SVD model...")
svd, R_pred = truncated_svd_predict(R_train, n_components=20)

print("Saving artifacts...")
os.makedirs("artifacts", exist_ok=True)
pickle.dump(user2idx, open("artifacts/user2idx.pkl", "wb"))
pickle.dump(item2idx, open("artifacts/item2idx.pkl", "wb"))
pickle.dump(R_pred, open("artifacts/R_pred.pkl", "wb"))
print("Artifacts saved in artifacts/ folder.")
