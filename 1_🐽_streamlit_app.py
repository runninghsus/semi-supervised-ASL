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

# # sidebar menu --> I just find this looks cool on line
# selected = option_menu(
#         menu_title = "Semi-Supervised Learning Of American Sign Language",
#         options = ["Main Menu", "Video Upload", "Annotate Sign-Language", "Prediction", "hello", "world"],
#         icons = ["house", "camera-video", "book", "pin", "bar-chart", "bag"],
#         menu_icon = "cast",
#         default_index = 0,
#         orientation = "horizontal",
#     )

current_page = utils.get_current_route()



# st.write(current_page)
if current_page == 'main menu':
    mainmenu.load_view()

elif current_page == 'video upload':
    videoUpload.load_view()
elif current_page == 'annotate':
    annotateSignLanguage.load_view()
elif current_page == 'prediction':
    prediction.load_view()
# elif selected ==''


# #original
# def main_page():
#     st.markdown("# Semi-Supervised Learning Of American Sign Language")
#     st.sidebar.markdown("# Semi-Supervised Learning Of American Sign Language")
#
# def videoUpload():
#     st.markdown("# videoUpload")
#     st.sidebar.markdown("# videoUpload")
#
# def annotateSignLanguage():
#     st.markdown("# annotateSignLanguage")
#     st.sidebar.markdown("# annotateSignLanguage")
#
# def prediction():
#     st.markdown("# prediction")
#     st.sidebar.markdown("# prediction")
#
# page_names_to_funcs = {
#     "Semi-Supervised Learning Of American Sign Language": main_page,
#     "videoUpload": videoUpload,
#     "annotateSignLanguage": annotateSignLanguage,
#     "prediction": prediction,
# }
#
# selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
# page_names_to_funcs[selected_page]()
