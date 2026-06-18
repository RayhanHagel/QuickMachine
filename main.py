import streamlit as st
import os




st.session_state.page_nav = {
    "data_create": os.path.join("pages", "data_create.py"),
    "config_create": os.path.join("pages", "config_create.py"),
    "config_run": os.path.join("pages", "config_run.py")
}


pages = {
    "Data": [
        st.Page(st.session_state.page_nav["data_create"], title="Data Create")
    ],
    "Config": [
        st.Page(st.session_state.page_nav["config_create"], title="Create Config"),
        st.Page(st.session_state.page_nav["config_run"], title="Config Run")
    ]
}

    
pg = st.navigation(pages, position="top")
pg.run()