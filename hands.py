import base64
import functools
from pathlib import Path
from steps import home, pose_extract, convnet_training, convnet_predict
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def img_to_html(img_path, width=500):
    img_html = f"<img src='data:image/png;base64,{img_to_bytes(img_path)}'  width='{width}px', class='img-fluid'>"
    return img_html

banner = './images/banner.png'
icon = './images/logo.png'
icon_img = Image.open(icon)
st.set_page_config(layout="wide",
                   page_title='SignWave',
                   page_icon=icon_img)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown(
    """
    <style>
    [data-testid="stSidebar"]{
        min-width: 250px;
        max-width: 250px;   
    }
    [data-testid="stSidebar"][aria-expanded="false"] {
        margin-left: -250px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.write('<style>div.block-container{padding-top:0rem;}</style>', unsafe_allow_html=True)


logo_placeholder = st.empty()
st.write('')
st.write('')
_, mid, _ = st.columns([1, 4, 1])


with st.sidebar:
    st.markdown("<p style='text-align: center; color: grey; '>" + img_to_html(banner, width=200) + "</p>",
                unsafe_allow_html=True)
    st.markdown(f" <h1 style='text-align: center; color: #67286D; font-size:18px; "
                f"font-family:Avenir ;font-weight:normal;'>Hello, {Path.home()}!</h1> "
                , unsafe_allow_html=True)

    app_names = np.array(['index',
                          'videoUpload',
                          'annotateSignLanguage',
                          'prediction',
                          ])
    menu_options = ['Menu',
                    'Video Upload',
                    'Sign Training',
                    'Predict',
                    ]
    icon_options = ['window-desktop',
                    'upload',
                    'diagram-2',
                    'file-earmark-plus',
                    ]

    selected = option_menu(None, ['Home', 'Upload videos', 'Learn signs', 'Predict signs'],
                           icons=['window-desktop', 'person-video', 'gpu-card', 'pencil-square'],
                           menu_icon="cast", default_index=0, orientation="vertical",
                           styles={
                               "container": {"padding": "0!important",
                                             "background-color": "#F0F2F6",
                                             },
                               "icon": {"color": "black",
                                        "font-size": "20px"},
                               "nav-link": {
                                   "font-size": "15px",
                                   "text-align": "left",
                                   "margin": "2px",
                                   "color": 'black',
                                   "--hover-color": "#F0F2F6"},
                               "nav-link-selected": {
                                   "font-weight": "normal",
                                   "color": "#F0F2F6",
                                   "background-color": '#67286D'
                               }
                           }
                           )


def navigation():
    if selected == 'Home':
        home.main()
    elif selected == 'Upload videos':
        pose_extract.main()
    elif selected == 'Learn signs':
        convnet_training.main()
    elif selected == 'Predict signs':
        convnet_predict.main()
    elif selected == None:
        home.load_view()
navigation()



