import streamlit as st
import pandas as pd
import time
import timeit


def app():
    column_data = pd.read_csv('data/meta/column_data.csv')
    column = column_data['column'][0]
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.preprocessing import normalize
    import numpy as np
    data = pd.read_csv('data/main_data.csv')
    
    st.subheader('Text')
    st.write(data)
    if st.button('Ubah data ke TF'):
        with st.spinner('tunggu sebentar ...'):
            time.sleep(2)
            st.subheader('TF')
            # calc TF vector
            start = timeit.default_timer()
            from sklearn.feature_extraction.text import CountVectorizer
            vectorizer = CountVectorizer()
            TF_vector = vectorizer.fit_transform(data[column])
            df_TF_vector = pd.DataFrame(TF_vector.toarray(),columns=vectorizer.get_feature_names_out())
            st.write(df_TF_vector)
            pd.DataFrame(df_TF_vector).to_csv('data/df_TF_vector.csv',index=False)
            stop = timeit.default_timer()
        waktu = stop-start
        st.write('lama proses: ', waktu, ' detik')
        st.success('Fitur berhasil diekstrak, silakan pergi ke proses selanjutnya')
        
            
    
    # tokenizer = tfidf.build_tokenizer() 
    # st.subheader('Cari kata pada dokumen')
    # feature_select = st.selectbox('Pilih fitur atau kata :',options=tfidf.get_feature_names(),key='feature_list')
    # doc_list = st.number_input('Pilih dokumen ke berapa (dari index ke-0):',min_value=0,max_value=99,key='doc_list')
    # count_token = tokenizer(data[column][doc_list]).count(feature_select)
    # len_doc = len(tokenizer(data[column][doc_list]))
    # st.write("Di dokumen ", doc_list ," " , feature_select ," ada ", count_token ," kata " , "dari ", len_doc, " kata")




    

