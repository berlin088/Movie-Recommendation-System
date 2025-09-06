import streamlit as st
import pickle
import numpy as np

st.title("Recommendation System Demo")

user_input = st.text_input("Enter userId (e.g. 1)")

if st.button("Recommend"):
    try:
        user2idx = pickle.load(open("artifacts/user2idx.pkl", "rb"))
        item2idx = pickle.load(open("artifacts/item2idx.pkl", "rb"))
        R_pred = pickle.load(open("artifacts/R_pred.pkl", "rb"))
        idx2item = {v: k for k, v in item2idx.items()}

        uid = int(user_input)
        if uid not in user2idx:
            st.error("User ID not found.")
        else:
            uidx = user2idx[uid]
            scores = R_pred[uidx]
            topk = np.argsort(scores)[-5:][::-1]
            item_ids = [idx2item[i] for i in topk]
            st.success(f"Top-5 recommended items for User {uid}: {item_ids}")
    except Exception as e:
        st.error(f"Error: {e}")
