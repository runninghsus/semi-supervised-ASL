import streamlit as st


def main():
    st.markdown(f" <h1 style='text-align: left; color: #67286D; font-size:30px; "
                f"font-family:Avenir; font-weight:normal'>Welcome to SignWave</h1> "
                , unsafe_allow_html=True)
    st.divider()
    st.markdown(f" <h1 style='text-align: left; color: #67286D; font-size:18px; "
                f"font-family:Avenir; font-weight:normal'>"
                f"SignWave is an innovative platform that utilizes advanced computer vision "
                f"and machine learning technologies to enable users to easily upload and analyze sign language videos. "
                f"With SignWave, users can add annotations, "
                f"apply convolutional neural networks to read sign language from 3D hand poses,"
                f"and batch predict new video files, all through a simple and intuitive no-code interface. "
                f"The name SignWave emphasizes the fluid and dynamic nature of hand gestures in sign language, "
                f""
                f"</h1> "
                , unsafe_allow_html=True)

    bottom_cont = st.container()
    with bottom_cont:
        st.divider()
        st.markdown(f" <h1 style='text-align: left; color: gray; font-size:16px; "
                    f"font-family:Avenir; font-weight:normal'>"
                    f"SignWave is developed by Alexander Hsu and Lucia Fang</h1> "
                    , unsafe_allow_html=True)