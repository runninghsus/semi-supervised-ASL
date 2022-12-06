# Contents of ~/my_app/streamlit_app.py
import streamlit as st
from pages_ import mainmenu, videoUpload, annotateSignLanguage, prediction

import utils

st.set_page_config(page_title='SignLanguage', page_icon='🖐️',
                   layout="wide", initial_sidebar_state="auto", menu_items=None)

st.set_option('deprecation.showPyplotGlobalUse', False)
utils.inject_custom_css()
utils.navbar_component()

reduce_header_height_style = """
    <style>
        div.block-container {padding-top:3rem;}
        div.block-container {padding-left:2rem;}
        div.block-container {padding-right:1rem;}
    </style>
"""
st.markdown(reduce_header_height_style, unsafe_allow_html=True)

current_page = utils.get_current_route()

if current_page == 'main menu':
    mainmenu.load_view()
elif current_page == 'video upload':
    videoUpload.load_view()
elif current_page == 'annotate':
    annotateSignLanguage.load_view()
elif current_page == 'prediction':
    prediction.load_view()

