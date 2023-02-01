import streamlit as st

from hands import swap_app

import categories

CATEGORY = categories.PREDICT
TITLE = "Translate"


def main():
    st.subheader("Sign Language Translation")
    # st.sidebar.markdown("# Prediction")
