import streamlit as st
import pandas as pd
import time
import timeit

def text_preprocessing(data,step,column='column'):
    col1, col2 = st.columns(2)
    with col1 :
        
        st.subheader('Data sebelum di proses')
        st.write(data)
        if step == "filtering" :
            options = st.multiselect('Pilih proses',
            ['hapus angka', 'hapus single karakter', 'hapus tanda baca'],
            ['hapus tanda baca'])    
    with col2 :
        with st.spinner('tunggu sebentar ...'):
            time.sleep(2)
            if step == 'case fold':
                start = timeit.default_timer()
                data[column] = data[column].str.lower()
                stop = timeit.default_timer()
            if step == "filtering" :
                start = timeit.default_timer()
                for option in options:
                    if option == 'hapus angka':
                        data[column] = data[column].str.replace('\d+', '', regex=True)
                    if option == 'hapus single karakter':
                        data[column] = data[column].str.replace("\b[a-zA-Z]\b", "", regex=True)
                    if option == "hapus tanda baca":
                        data[column] = data[column].str.replace('[^\w\s]+', '', regex=True)
                stop = timeit.default_timer()
            if step == 'remove stopwords' :
                start = timeit.default_timer()
                import nltk
                nltk.download('punkt')
                from nltk.tokenize import word_tokenize
                from nltk.corpus import stopwords
                nltk.download('stopwords')
                data[column] = data[column].apply(word_tokenize)
                data[column] =data[column].apply(lambda x: [token for token in x if token not in stopwords.words('indonesian')])
                data[column] = data[column].str.join(" ")
                stop = timeit.default_timer()
            if step == 'stemming' :
                start = timeit.default_timer()
                from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
                factory = StemmerFactory()
                stemmer = factory.create_stemmer()
                data[column] = data[column].str.split()
                data[column] = data[column].apply(lambda x: [stemmer.stem(y) for y in x]) # Stem every word.
                data[column] = data[column].str.join(" ")
                stop = timeit.default_timer()
            data.to_csv('data/data_branch.csv',index=False)
            data_branch = pd.read_csv('data/data_branch.csv')
            st.subheader('Data setelah di proses')
            st.write(data_branch)
            waktu = stop-start
            st.write('lama proses : ', waktu,' detik')
            df_preprocessing = pd.read_csv('data/meta/data_preprocessing.csv')
            df_preprocessing.loc[df_preprocessing["steps"] == step, "time"] = waktu
            if st.button('simpan data'):
                data_branch.to_csv('data/main_data.csv',index=False)
                df_preprocessing.to_csv('data/meta/data_preprocessing.csv',index=False)
                st.success('Berhasil disimpan')
def app():
    #Your statements here
    data = pd.read_csv('data/main_data.csv')
    # data = pd.read_csv('data/main_data_sidang.csv')
    data.columns = data.columns.str.replace("^\s+|\s+$","",regex=True)
    column_data = pd.read_csv('data/meta/column_data.csv')
    column = column_data['column'][0]

    st.sidebar.markdown("lakukan preprosesing data")
    steps = st.sidebar.radio('Langkah langkah pre-prosessing : ',('case fold','filtering','remove stopwords','stemming'))
    st.header(f'Preprosesing - {steps}')
    if steps == 'case fold':
        text_preprocessing(data,'case fold',column)
    if steps == 'filtering':
        text_preprocessing(data,'filtering',column)
    if steps == "remove stopwords":
        text_preprocessing(data,'remove stopwords',column)
    if steps == 'stemming' :
        text_preprocessing(data, 'stemming',column)
