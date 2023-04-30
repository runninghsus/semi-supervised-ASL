import streamlit as st


def main():
    # page title
    st.markdown(f" <h1 style='text-align: left; color: #67286D; font-size:30px; "
                f"font-family:Avenir; font-weight:normal'>"
                f"Upload new videos, with signs decoded."
                f""
                f"</h1> "
                , unsafe_allow_html=True)
    st.divider()
