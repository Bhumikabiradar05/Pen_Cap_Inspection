import streamlit as st
import pandas as pd
from datetime import datetime

def init_review_queue():
    if "review_queue" not in st.session_state:
        st.session_state.review_queue = []

def add_to_queue(image, score, status):
    init_review_queue()
    item = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image": image,
        "score": score,
        "status": status,
        "decision": "Pending"
    }
    st.session_state.review_queue.append(item)