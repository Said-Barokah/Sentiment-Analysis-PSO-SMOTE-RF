import streamlit as st
import pandas as pd
import time
import os
import timeit

def app():
    st.title('APLIKASI SENTIMEN ANALASIS')
    if (os.path.exists("data/data_master.csv")):
         st.text('Data master')
         df = pd.read_csv('data/data_master.csv')
         st.write(df)
    if (os.path.exists("data/meta/column_data.csv")):
         column = pd.read_csv('data/meta/column_data.csv')
         feature = column['column'][0]
         label  = column['label'][0]
    data = st.file_uploader("upload data berformat csv (untuk mengubah data master)", type=['csv'])
    if data is not None:
            dataframe = pd.read_csv(data)
            dataframe.columns = dataframe.columns.str.replace("^\s+|\s+$","",regex=True)
            st.write(dataframe)
            col1, col2 = st.columns(2)
            with col1 :
                column = st.selectbox("Pilih Kolom yang akan di proses :",
                list(dataframe.columns))
            with col2 :
                label = st.selectbox("Pilih Kolom yang akan dijadikan label atau class :",
                list(dataframe.columns))

            column_data = pd.DataFrame(data={'column': [column], 'label': [label]})
            if st.button('simpan data') :
                start = timeit.default_timer()
                dataframe[label] = dataframe[label].str.replace("^\s+|\s+$","",regex=True)
                dataframe.to_csv('data/data_master.csv',index=False)
                column_data.to_csv('data/meta/column_data.csv', index=False)
                dataframe = dataframe[[column_data['column'][0],column_data['label'][0]]]
                dataframe.to_csv('data/main_data.csv',index=False)
                # dataframe.to_csv('data/main_data_sidang.csv',index=False)
                with st.spinner('tunggu sebentar ...'):
                    time.sleep(1)
                st.text(f'kolom yang digunakan dalam klasifikasi adalah "{feature}" dan kolom yang akan dijadikan label adalah "{label}"')
                st.success('data berhasil disimpan')
                stop = timeit.default_timer()
                waktu = stop-start
                st. write('lama proses :', waktu, ' detik')
                st.info('column ' + column_data['column'][0] + ' akan diproses')
                st.info('column ' + column_data['label'][0] + ' akan dijadikan label')