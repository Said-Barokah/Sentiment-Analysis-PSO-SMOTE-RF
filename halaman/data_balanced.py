def app():
    from imblearn.over_sampling import SMOTEN
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import streamlit as st
    import timeit
    
    data = pd.read_csv('data/main_data.csv')
    data[data.columns[1]] = data[data.columns[1]].str.replace("^\s+|\s+$","",regex=True)
    df_TF_vector = pd.read_csv('data/df_TF_vector.csv')
    column_data = pd.read_csv('data/meta/column_data.csv')
    test_size = pd.read_csv('data/meta/test_size.csv')


    data_train, data_test, data_y_train, data_y_test = train_test_split(data, data[column_data['label'][0]], test_size = test_size['test size'][0],random_state=1587)
    X_train, X_test, y_train, y_test = train_test_split(df_TF_vector, data[column_data['label'][0]], test_size = test_size['test size'][0],random_state=1587)
    sampler = SMOTEN(sampling_strategy='minority',k_neighbors=5,random_state=1587)
    X_res, y_res = sampler.fit_resample(X_train, y_train)
    st.subheader('Data Train')
     
    
    tab1, tab2 = st.tabs(["Text", "TF"])
    with tab1:
        st.write(data_train)
    with tab2:
        st.write(X_train)

    import matplotlib.pyplot as plt
    fig2,ax2 = plt.subplots(figsize=(7,8))
    xs_1 = data_y_train.unique()
    ys_1 = data_y_train.value_counts()
    st.write(xs_1)
    st.write(ys_1)
    ax2.bar(xs_1, ys_1)
    for x,y in zip(xs_1,ys_1):
        ax2.annotate(text=y,xy=(x,y),textcoords="offset points", xytext=(0,10),ha="center")
    ax2.set_title('Dataset sebelum oversampling SMOTE')
    st.pyplot(fig2)


    expander = st.expander(f"Hasil Penyeimbangan Data dari data train")
    start = timeit.default_timer()
    from sklearn.feature_extraction.text import CountVectorizer
    from scipy.sparse import csr_matrix
    count_vectorizer = CountVectorizer()
    count_vectorizer.fit_transform(X_train)
    text = count_vectorizer.inverse_transform(X_res)
    text = [' '.join(row) for row in text]
    df_sintetis = pd.DataFrame(text, columns=['data sintetis'])
    df_sintetis_full = df_sintetis.join(y_res)
    df_sintetis_add = df_sintetis[X_train.shape[0]:].join(y_res[X_train.shape[0]:])
    X_sintetis_add = X_res.iloc[X_train.shape[0]:]
    tab3, tab4 = expander.tabs(["Text","TF"])
    with tab3:
        st.write(df_sintetis_add)
        df_sintetis_full.to_csv('data/df_sintetis_text_train.csv',index=False)
    with tab4:
        st.write(X_sintetis_add)
        X_res.to_csv('data/df_sintetis_num_train.csv',index=False)

    fig, ax = plt.subplots(figsize=(7,8))
    xs = y_res.unique()
    ys = y_res.value_counts()
    ax.bar(xs, ys)
    for x,y in zip(xs,ys):
        ax.annotate(text=y,xy=(x,y),textcoords="offset points", xytext=(0,10), ha='center')
    ax.set_title('Dataset setelah oversampling SMOTE')
    expander.pyplot(fig)
    stop = timeit.default_timer()
    waktu = stop-start
    st.write('lama proses : ', waktu, ' detik')


    


