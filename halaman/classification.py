import streamlit as st
import pandas as pd
import timeit

def app() :
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    #penambahan 
    data_master = pd.read_csv('data/data_master.csv')
    data_main = pd.read_csv('data/main_data.csv')
    data_sin = pd.read_csv('data/df_sintetis_text_train.csv')
    df_TF_vector_sin = pd.read_csv('data/df_sintetis_num_train.csv')
    df_TF_vector = pd.read_csv('data/df_TF_vector.csv')
    test_size = pd.read_csv('data/meta/test_size.csv')
    column_data = pd.read_csv('data/meta/column_data.csv')
    column_data = pd.read_csv('data/meta/column_data.csv')
    param_pso = pd.read_csv('data/meta/detail_parameter_pso.csv')
    # pso_fea = pd.read_csv('data/meta/selected_feature.csv')
    
    acc_dict = {'model':[],'akurasi':[]}
    time_dict = {'model':[],'waktu':[]}
    df_param_pso_t = param_pso.T

    
    # st.sidebar.write(df_param_pso_t)
    # pso_fea = pd.read_csv('data\meta\Populasi\selected_40_10.csv')
    # pso_fea = pd.read_csv('data\meta\Populasi\selected_40_20.csv')
    # pso_fea = pd.read_csv('data\meta\Populasi\selected_40_30.csv')
    pso_fea = pd.read_csv('data\meta\Populasi\selected_40_40.csv')
    # pso_fea = pd.read_csv('data\meta\Populasi\selected_40_50.csv')
    # pso_fea = pd.read_csv('data\meta\Populasi\selected_40_60.csv')
    # pso_fea = pd.read_csv('data\meta\Populasi\selected_40_70.csv')
    # pso_fea = pd.read_csv('data\meta\Populasi\selected_40_80.csv')
    # pso_fea = pd.read_csv('data\meta\Populasi\selected_40_90.csv')
    # pso_fea = pd.read_csv('data\meta\Populasi\selected_40_100.csv')
    # pso_fea = pd.read_csv('data\meta\Populasi 2\selected_40_10_2.csv')
    # pso_fea = pd.read_csv('data\meta\Populasi 2\selected_40_20_2.csv')
    # pso_fea = pd.read_csv('data\meta\Populasi 2\selected_40_30_2.csv')
    # pso_fea = pd.read_csv('data\meta\Populasi 2\selected_40_40_2.csv')
    # pso_fea = pd.read_csv('data\meta\Populasi 2\selected_40_50_2.csv')
    # pso_fea = pd.read_csv('data\meta\Populasi 2\selected_40_60_2.csv')
    # pso_fea = pd.read_csv('data\meta\Populasi 2\selected_40_70_2.csv')
    # pso_fea = pd.read_csv('data\meta\Populasi 2\selected_40_80_2.csv')
    # pso_fea = pd.read_csv('data\meta\Populasi 2\selected_40_90_2.csv')
    # pso_fea = pd.read_csv('data\meta\Populasi 2\selected_40_100_2.csv')

    st.subheader('Klasifikasi Random Forest')
    
    from sklearn.metrics import accuracy_score, confusion_matrix
    from mlxtend.plotting import plot_confusion_matrix
    
    X_train, X_test,y_train, y_test = train_test_split(df_TF_vector, data_main[column_data['label'][0]], test_size=test_size['test size'][0], random_state=1587)
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(max_depth=5, random_state=1587, n_estimators=200)
    

    # clf_x_train_pso = df_TF_vector_sin[list(map(str,list(pso_fea['Selected Features'])))]
    # clf_y_train_pso = data_sin[column_data['label'][0]]
    clf_x_test_pso = X_test[list(map(str,list(pso_fea['Selected Features'])))]
   
   #Buat yang No SMOTE
    clf_x_train_pso = X_train[list(map(str,list(pso_fea['Selected Features'])))]
    clf_y_train_pso = y_train


    start = timeit.default_timer()
    clf_train = clf.fit(clf_x_train_pso,clf_y_train_pso)
    y_pred = clf_train.predict(clf_x_test_pso)
    stop = timeit.default_timer()

    time = stop-start 
    accuracy = accuracy_score(y_test,y_pred)
    c_matrik = confusion_matrix(y_test,y_pred)


    expander_1 = st.expander("Hasil dari Klasifikasi Random Forest + PSO + SMOTE")
    column = expander_1.multiselect('Pilih kolom yang akan di digunakan, kecuali kolom label',
                list(data_master.columns), key=1
                )
    data_pred = pd.DataFrame({'Prediksi Kelas':y_pred,'Kelas Sesunggunya':y_test})
    data_pred = (data_pred.join(data_master[column]))
    expander_1.write(data_pred)
    expander_1.write(f'akurasi yang didapatkan yaitu : {round(accuracy,3)}')
    fig, ax = plot_confusion_matrix(conf_mat=c_matrik,class_names=clf_train.classes_)
    ax.set_title('Confusion Matrik')
    ax.set_ylabel('Actual Label')
    ax.set_xlabel('Predicted Label')
    expander_1.pyplot(fig)
    acc_dict['akurasi'].append(round(accuracy,3))
    acc_dict['model'].append('Random Forest + PSO + SMOTE')
    time_dict['waktu'].append(time)
    time_dict['model'].append('Random Forest + PSO + SMOTE')

    expander_2 = st.expander("Hasil dari Klasifikasi Random Forest + SMOTE")
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.ensemble import RandomForestClassifier
    from mlxtend.plotting import plot_confusion_matrix
    
    clf = RandomForestClassifier(max_depth=5, random_state=1587, n_estimators=200)
    

    clf_x_train = df_TF_vector_sin
    clf_y_train = data_sin[column_data['label'][0]]

    #HANYA RF
    # clf_x_train = X_train
    # clf_y_train = y_train[column_data['label'][0]]


    start = timeit.default_timer()
    clf_train = clf.fit(clf_x_train,clf_y_train )

    # This may not the best way to view each estimator as it is small
    from sklearn import tree
    import matplotlib.pyplot as plt

    # ini untuk index ke 0
    fig, axes = plt.subplots(figsize=(10, 5), dpi=1500)
    tree.plot_tree(clf.estimators_[0],
                   class_names=clf_train.classes_,
                   feature_names=clf_train.feature_names_in_
                   )
    # ini untuk memebrikan judul
    axes.set_title('Estimator: ' + str(0), fontsize=11)
    # ini untuk menyimpan gambar
    fig.savefig('rf_0_trees.png')
    st.write(fig)

    y_pred = clf_train.predict(X_test)
    stop = timeit.default_timer()
    time = stop-start 

    accuracy = accuracy_score(y_test,y_pred)
    c_matrik = confusion_matrix(y_test,y_pred)
    column = expander_2.multiselect('Pilih kolom yang akan di digunakan, kecuali kolom label',
                list(data_master.columns),key=2
                )
    data_pred = pd.DataFrame({'Predsi Kelas':y_pred,'Kelas Sesunggunya':y_test})
    data_pred = (data_pred.join(data_master[column]))
    expander_2.write(data_pred)
    expander_2.write(f'akurasi yang didapatkan yaitu : {round(accuracy,3)}')
    fig, ax = plot_confusion_matrix(conf_mat=c_matrik,class_names=clf_train.classes_)
    ax.set_title('Confusion Matrik')
    ax.set_ylabel('Actual Label')
    ax.set_xlabel('Predicted Label')
    expander_2.pyplot(fig)
    acc_dict['akurasi'].append(round(accuracy,3))
    acc_dict['model'].append('Random Forest + SMOTE')
    time_dict['waktu'].append(time)
    time_dict['model'].append('Random Forest + SMOTE')


    st.subheader('Perbandingan nilai akurasi')
    tab1, tab2 = st.tabs(['akurasi','waktu'])
    with tab1:
        df_acc = pd.DataFrame(acc_dict['akurasi'],index=acc_dict['model'])
        df_acc.to_csv('data/meta/akurasi 2/akurasi_40_40_2.csv')
        st.bar_chart(df_acc)
    with tab2:
        df_time = pd.DataFrame(time_dict['waktu'],index=acc_dict['model'])
        df_time.to_csv('data/meta/waktu 2/waktu_40_40_2.csv')
        st.bar_chart(df_time)


    

    


