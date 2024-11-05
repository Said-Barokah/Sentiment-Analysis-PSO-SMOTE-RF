import streamlit as st
import pandas as pd

from halaman import upload_data,pre_processing,fea_selection,fea_extraction,data_split,data_balanced,classification
pag_name = {
    "Upload Data" : upload_data.app,
    "Pre Processing" : pre_processing.app,
    "Feature Extraction" : fea_extraction.app,
    "Feature Selection":fea_selection.app,
    "Data Split":data_split.app,
    "Data Balanced":data_balanced.app,
    "Classification" : classification.app
}

demo_name = st.sidebar.selectbox("halaman", pag_name.keys())
pag_name[demo_name]()