import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
import random
import timeit

from niapy.problems import Problem
from niapy.task import Task
from niapy.algorithms.basic import ParticleSwarmOptimization


class RFFeatureSelection(Problem):
    def __init__(self, X_train, y_train,X_test,y_test, alpha=0.99):
        super().__init__(dimension=X_train.shape[1], lower=0, upper=1)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.alpha = alpha

    def _evaluate(self, x):
        selected = x == 1.0
        num_selected = selected.sum()
        if num_selected == 0:
            return 1.0
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        
        clf = RandomForestClassifier(max_depth=2, random_state=0)
        # st.write(self.X_train)
        index_selected = np.where(selected)
        clf_train = clf.fit(self.X_train.iloc[:, index_selected[0]], self.y_train)
        y_pred = clf_train.predict(self.X_test.iloc[:, index_selected[0]])
        accuracy = accuracy_score(self.y_test,y_pred)
        # st.write(f"index fitur terpilih {index_selected[0]}")
        # st.write(f'Akurasi {accuracy}')
        score = 1 - accuracy
        num_features = self.X_train.shape[1]
        return self.alpha * score + (1 - self.alpha) * (num_selected / num_features)
def app():
    from sklearn.datasets import load_breast_cancer
    from niapy.task import Task
    from niapy.algorithms.basic import ParticleSwarmOptimization
    from sklearn.model_selection import train_test_split, cross_val_score
    
    st.header('Particle Swarm Optimization')


    random_value = random.random()

    population_size = st.number_input('population size', value=10)
    c1 = st.number_input('c1 (bobot kognitif)',value=0.7)
    c2 = st.number_input('c2 (bobot sosial)',value=0.5)
    w = st.number_input('w (bobot inertia)',value=random_value)
    min_velocity = st.number_input('minimal kecepatan',value=0)
    max_velocity = st.number_input('maksimal kecepatan',value=1)
    iterasi = st.number_input('jumlah generasi',value=40)
    

    dict_par_pso = {
        'population_size' : population_size,
        'c1' : c1,
        'c2' : c2,
        'w' : w,
        'min_velocity' : min_velocity,
        'max_velocity' : max_velocity,
        'iterasi'   : iterasi
    }

    df_param_pso = pd.DataFrame([dict_par_pso])
    df_param_pso.to_csv('data/meta/detail_parameter_pso.csv',index=False)
    data = pd.read_csv('data/main_data.csv')
    df_TF_vector = pd.read_csv('data/df_TF_vector.csv')
    column_data = pd.read_csv('data/meta/column_data.csv')
    test_size = pd.read_csv('data/meta/test_size.csv')
    X_train, X_test, y_train, y_test = train_test_split(df_TF_vector, data[column_data['label'][0]], test_size = test_size['test size'][0],random_state=1587)

    problem = RFFeatureSelection(X_train, y_train,X_test,y_test)
    task = Task(problem, max_iters=iterasi)
    algorithm = ParticleSwarmOptimization(population_size=population_size, seed=1234, c1=c1, c2=c2, w=w, min_velocity =min_velocity, max_velocity=max_velocity)
    with st.spinner('tunggu sebentar ...'):
        best_features, best_fitness = algorithm.run(task)
    selected_features = np.where(best_features == 1)
    selected_features_df = pd.DataFrame(df_TF_vector.columns[selected_features], columns=["Selected Features"])
    st.caption(f'Nilai fitnes terbaik yang didapatkan {round(best_fitness,3)} dari subset fitur terpilih dibawah ini')
    st.dataframe(selected_features_df)
    selected_features_df.to_csv('data/meta/selected_feature.csv',index=False)
    st.success('Fitur berhasil terseleksi')







   

    
    
