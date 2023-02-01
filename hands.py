import configparser as cfg
import functools
import tempfile
from pathlib import Path

import hydralit_components as hc
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu

import steps as _stable_apps
import categories as _categories
import utilities
# from utils.load_workspace import load_features, load_iterX
# from config.help_messages import UPLOAD_CONFIG_HELP, IMPRESS_TEXT


def get_url_app():
    """
    :return: current app key
    """
    try:
        return st.experimental_get_query_params()["app"][0]
    except KeyError:
        return "index"


def swap_app(app):
    """
    :param app: the desired app to toggle to
    :return:
    """
    st.experimental_set_query_params(app=app)
    session_state = utilities.session_state()
    if not session_state.app == app:
        session_state.app = app
        # time.sleep(0.1)
        st.experimental_rerun()


def index(application_options):
    """
    :param application_options: dictionary (app_key: apps)
    :return:
    """
    num_columns = len(_categories.APPLICATION_CATEGORIES_BY_COLUMN.keys())
    columns = st.columns(num_columns)
    bottom_cont = st.container()
    HERE = Path(__file__).parent.resolve()

    step1_fname = HERE.joinpath("images/mediapipe_schematic.png")
    step1_im = Image.open(step1_fname)
    step2_fname = HERE.joinpath("images/sign_letters.png")
    step2_im = Image.open(step2_fname)
    step3_fname = HERE.joinpath("images/translate_logo.png")
    step3_im = Image.open(step3_fname)
    step4_fname = HERE.joinpath("images/conv_net_example.png")
    step4_im = Image.open(step4_fname)

    # 4 column layout
    count = 1
    for (
            column_index,
            categories,
    ) in _categories.APPLICATION_CATEGORIES_BY_COLUMN.items():
        column = columns[column_index]
        # for each category
        for category in categories:
            applications_in_this_category = [
                item
                for item in application_options.items()
                if item[1].CATEGORY == category
            ]
            # create a container expander that puts an image/gif/video and a button to navigate

            for app_key, application in applications_in_this_category:
                    # app_ = column.expander(f'{category}', expanded=True)
                    app_ = column.expander(f'#{count}', expanded=True)
                    if application.TITLE == 'Video Upload':
                        app_.image(step1_im)
                    elif application.TITLE == 'Sign Training':
                        app_.image(step2_im)
                    elif application.TITLE == "Translate":
                        app_.image(step3_im)
                    # elif application.TITLE == 'Refine Behaviors':
                    #     app_.image(step4_im)
                    count += 1
                    if app_.button(f'{application.TITLE}', key=app_key):
                        swap_app(app_key)

    with bottom_cont:
        st.markdown("""---""")
        st.write('')
        # st.markdown('<span style="color:grey">{}</span>'.format(IMPRESS_TEXT), unsafe_allow_html=True)


def _get_apps_from_module(module):
    """
    :param module: a module folder that contains each python file app
    :return: list of apps
    """
    # strip _ and ignore __init__.py
    apps = {
        item.replace("_", "-"): getattr(module, item)
        for item in dir(module)
        if not item.startswith("_")
    }

    return apps


def main():
    session_state = utilities.session_state(app=get_url_app())
    stable_apps = _get_apps_from_module(_stable_apps)
    HERE = Path(__file__).parent.resolve()
    logo_fname_ = HERE.joinpath("images/hands_logo.jpeg")
    logo_im_ = Image.open(logo_fname_)
    #TODO: Decide on one, delete over
    logo_fname2 = HERE.joinpath("images/translate_logo.png")
    logo_im2 = Image.open(logo_fname2)

    # set webpage icon and layout
    st.set_page_config(
        page_title="Hands",
        # page_icon=logo_im_,
        page_icon=logo_im2,
        layout="wide",
        # initial_sidebar_state="expanded",
        menu_items={
        }
    )

    hide_streamlit_style = """
                <style>
                MainMenu {visibility: hidden;}
                footer {visibility: hidden;}


                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    st.set_option('deprecation.showPyplotGlobalUse', False)
    header_container = st.container()
    # locate module
    application_options = {**stable_apps}
    cc = st.columns(len(application_options))
    # if stuck in no mans land, return to index (main menu)
    if (
            session_state.app != "index"
            and not session_state.app in application_options.keys()
    ):
        print(f"Option {session_state.app} not valid. Returning to main menu...")
        swap_app("index")
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 250px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 250px;
            margin-left: -500px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    theme_bad = {'bgcolor': '#F0F2F6', 'title_color': 'orange', 'content_color': 'orange', 'icon_color': 'orange',
                 'icon': 'fa fa-question-circle'}
    theme_okay = {'bgcolor': '#F0F2F6', 'title_color': 'yellow', 'content_color': 'yellow', 'icon_color': 'yellow',
                  'icon': 'fa fa-question-circle'}
    theme_good = {'bgcolor': '#F0F2F6', 'title_color': 'green', 'content_color': 'green', 'icon_color': 'green',
                  'icon': 'fa fa-check-circle'}
    # if in main menu, display applications, see above index for item layout
    with st.sidebar:
        le, ri = header_container.columns([1, 1])

        # uploaded_config = le.file_uploader('upload config file'.upper()
        #                                    , type='ini'
        #                                    )
        #
        # if uploaded_config is not None:
        #     # Make temp file path from uploaded file
        #     with tempfile.NamedTemporaryFile(mode="wb", delete=False) as temp:
        #         bytes_data = uploaded_config.getvalue()
        #         temp.write(bytes_data)
        #     project_config = cfg.ConfigParser()
        #     project_config.optionxform = str
        #     with open(temp.name) as file:
        #         project_config.read_file(file)

        # try:
        #     sections = [x for x in project_config.keys() if x != "DEFAULT"]
        #     for parameter, value in project_config[sections[0]].items():
        #         if parameter == 'PROJECT_PATH':
        #             working_dir = value
        #         elif parameter == 'PROJECT_NAME':
        #             prefix = value
        #         elif parameter == 'CLASSES':
        #             annotations = value
        #     with cc[0]:
        #         hc.info_card(title='Upload Data',
        #                      content=f'{working_dir}/{prefix}', bar_value=100,
        #                      icon_size="1.5rem", title_text_size="1rem", content_text_size="0.8rem",
        #                      theme_override=theme_good, key='first')
        #     try:
        #         [features_train, _, _, _] = load_features(working_dir, prefix)
        #         with cc[1]:
        #             hc.info_card(title='Extract Features',
        #                          content=f'Extracted {len(features_train)} random splits, each with '
        #                                  f'features {[*np.vstack(features_train[0]).shape]} for '
        #                                  f'{annotations}', bar_value=100,
        #                          icon_size="1.5rem", title_text_size="1rem", content_text_size="0.8rem",
        #                          theme_override=theme_good, key='second')
        #     except:
        #         with cc[1]:
        #             hc.info_card(title='Extract Features', content='', bar_value=5,
        #                          icon_size="1.5rem", title_text_size="1rem", content_text_size="0.8rem",
        #                          theme_override=theme_bad, key='second')
        #     try:
        #         [_, _, _, scores, _, _] = load_iterX(working_dir, prefix)
        #         with cc[2]:
        #             rounded_scores = [int(100*round(np.mean(scores[-1], axis=0)[c], 2))
        #                               for c in range(len(scores[-1][0]))]
        #             classes = annotations.split(', ')
        #             hc.info_card(title='Active Learning',
        #                          content=f'Final classication was {[*rounded_scores]}% '
        #                                  f'for behaviors: {classes[:-1]}', bar_value=100,
        #                          icon_size="1.5rem", title_text_size="1rem", content_text_size="0.8rem",
        #                          theme_override=theme_good, key='third')
        #     except:
        #         with cc[2]:
        #             hc.info_card(title='Active Learning', content='', bar_value=5,
        #                          icon_size="1.5rem", title_text_size="1rem", content_text_size="0.8rem",
        #                          theme_override=theme_bad, key='third')
            # try:
            #     [predict, proba, outlier_indices] = load_predict_proba(working_dir, prefix)
            #     with cc[3]:
            #         # can just use 'good', 'bad', 'neutral' sentiment to auto color the card
            #         hc.info_card(title='Refine Behaviors', content='', sentiment='good', bar_value=60,
            #                      icon_size="1.5rem", title_text_size="1rem", content_text_size="0.8rem",
            #                      theme_override=theme_okay, key='fourth')
            # except:
            #     with cc[3]:
            #         hc.info_card(title='Refine Behaviors', content='', bar_value=5,
            #                      icon_size="1.5rem", title_text_size="1rem", content_text_size="0.8rem",
            #                      theme_override=theme_bad, key='fourth')

        # except:
        #     with cc[0]:
        #         hc.info_card(title='Upload Data', content='', bar_value=5,
        #                      icon_size="1.5rem", title_text_size="1rem", content_text_size="0.8rem",
        #                      theme_override=theme_bad, key='first')

        _, mid_im, _ = st.columns([0.05, 1, 0.05])
        st.image(logo_im_)
        _, mid_im2, _ = st.columns([0.215, 1, 0.215])

        # if mid_im2.button("Translate Signs"):
        #     swap_app('videoUpload')
        mid_im2.write('Sign Translation App')
        st.write('---')
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
        nav_options = option_menu(None, menu_options,
                                  icons=icon_options,
                                  menu_icon="",
                                  default_index=int(np.where(app_names == get_url_app())[0]),
                                  # orientation="horizontal",
                                  styles={
                                      "container": {"padding": "0!important",
                                                    "background-color": "#F0F2F6",
                                                    # "background-color": "#FFB7E5",
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
                                          # "text-decoration": "underline",
                                          "color": "#F0F2F6",
                                          "background-color": '#f6386d'
                                          # "color": "#f6386d",
                                          # "background-color": '#FFB7E5'
                                      },
                                  }
                                  )
        for i in range(len(menu_options)):
            if nav_options == menu_options[i]:
                swap_app(app_names[i])

    if session_state.app == "index":
        application_function = functools.partial(
            index, application_options=application_options,
        )

    else:
        try:
            application_function = functools.partial(
                application_options[session_state.app].main,
                # config=project_config,
            )
        except:
            application_function = functools.partial(
                application_options[session_state.app].main,
                # config=None,
            )

    application_function()


if __name__ == "__main__":
    main()
