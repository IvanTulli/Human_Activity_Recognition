import time, streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import warnings
warnings.filterwarnings('ignore')

import pickle

def main():
    st.title('Human Activity Classification via Smartphone Readings')
    st.sidebar.title('Multiclass Classification Web App')
    st.markdown('Determining whether someone is standing, laying, sitting, walking, walking downstairs, or walking upstairs.')

    @st.cache_data
    def load_data():
        df_train = pd.read_csv(r"C:\Git\Human_Activity_Recognition\Classification Problem\train.csv")
        df_test  = pd.read_csv(r"C:\Git\Human_Activity_Recognition\Classification Problem\test.csv")

        # drop subject column from features
        df_train = df_train.drop(columns=['subject'])
        X_train = df_train.iloc[:, :-1]
        y_train = df_train.iloc[:, -1]

        X_test  = df_test.drop(columns=['subject']).iloc[:, :-1]
        y_test  = df_test.iloc[:, -1]
        return X_train, X_test, y_train, y_test

    def plot_conf_matrix(y_true, y_pred, label_names):
        cm = confusion_matrix(y_true, y_pred, labels=label_names)
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay(cm, display_labels=label_names).plot(
            ax=ax, colorbar=True, xticks_rotation=90
        )
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix")
        st.pyplot(fig)

    X_train, X_test, y_train, y_test = load_data()
    labels = sorted(y_train.unique().tolist())   # explicit class order
    trainchoice= st.sidebar.selectbox("Make a choice:", ("-", "Train a new model", "Use a trained model"))
    if trainchoice == "Train a new model":
        clf_choice = st.sidebar.selectbox(
            'Classifier',
            ('Random Choice (Baseline)', 'Logistic Regression', 'Random Forest')
        )

        plot_cm_choice = st.sidebar.radio("Plot the confusion matrix?", ('Yes', 'No'))

        if clf_choice == 'Random Choice (Baseline)':
            if st.sidebar.button("Classify", key='classify_baseline'):
                st.subheader("Random Choice Results")

                rng = np.random.default_rng()
                # random labels drawn from the class set
                y_pred = rng.choice(labels, size=len(y_test))

                acc = accuracy_score(y_test, y_pred)
                report = classification_report(
                    y_test, y_pred, labels=labels, output_dict=True, zero_division=0
                )
                df = pd.DataFrame(report).transpose().round(3)

                st.write(f"Accuracy: **{acc:.3f}**")
                st.table(df.loc[labels, ['precision', 'recall', 'f1-score']])

                if plot_cm_choice == 'Yes':
                    plot_conf_matrix(y_test, y_pred, labels)

        if clf_choice == 'Logistic Regression':
            st.sidebar.subheader("Model Hyperparameters")
            C = st.sidebar.number_input("C (Regularization)", 0.01, 10.0, value=1.0, step=0.01, key='C_LR')

            if st.sidebar.button("Classify", key='classify_lr'):
                st.subheader("Logistic Regression Results")
                with st.spinner("Training model..."):
                
                    model = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(C=C, solver='lbfgs', multi_class='auto', max_iter=2000, n_jobs=None)
                    )
                    model.fit(X_train, y_train)
                    time.sleep(0.5)  
                    st.success("Training complete!")
                y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                report = classification_report(
                    y_test, y_pred, labels=labels, output_dict=True, zero_division=0
                )
                df = pd.DataFrame(report).transpose().round(3)

                st.write(f"Accuracy: **{acc:.3f}**")
                st.table(df.loc[labels, ['precision', 'recall', 'f1-score']])

                if plot_cm_choice == 'Yes':
                    plot_conf_matrix(y_test, y_pred, labels)

        if clf_choice == 'Random Forest':
            st.sidebar.subheader("Model Hyperparameters")
            n_estimators = st.sidebar.slider("n_estimators", 50, 500, 200, step=50, key='n_rf')
            max_depth    = st.sidebar.slider("max_depth (0 = None)", 0, 50, 0, step=1, key='d_rf')

            if st.sidebar.button("Classify", key='classify_rf'):
                st.subheader("Random Forest Results")
                with st.spinner("Training model..."):
                    rf = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=None if max_depth == 0 else max_depth,
                        random_state=42,
                        n_jobs=-1
                    )
                    rf.fit(X_train, y_train)
                    st.success("Training Complete!")
                y_pred = rf.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                report = classification_report(
                    y_test, y_pred, labels=labels, output_dict=True, zero_division=0
                )
                df = pd.DataFrame(report).transpose().round(3)

                st.write(f"Accuracy: **{acc:.3f}**")
                st.table(df.loc[labels, ['precision', 'recall', 'f1-score']])

                if plot_cm_choice == 'Yes':
                    plot_conf_matrix(y_test, y_pred, labels)
    elif trainchoice == "Use a trained model":
        clf_choice = st.sidebar.selectbox('Choose trained classifier', ('Logistic Regression', 'Random Forest'))
        plot_cm_choice = st.sidebar.radio("Plot the confusion matrix?", ('Yes', 'No'))
        if clf_choice == "Logistic Regression":
            with open("lr_model.pkl", "rb") as f:
                loaded_lr = pickle.load(f)
            if st.sidebar.button("Predict", key = "predict_lr"):
                y_pred = loaded_lr.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                report = classification_report(
                        y_test, y_pred, labels=labels, output_dict=True, zero_division=0)
                df = pd.DataFrame(report).transpose().round(3)

                st.write(f"Accuracy: **{acc:.3f}**")
                st.table(df.loc[labels, ['precision', 'recall', 'f1-score']])

                if plot_cm_choice == 'Yes':
                    plot_conf_matrix(y_test, y_pred, labels)

        if clf_choice == "Random Forest":
            with open("rf_model.pkl", "rb") as f:
                loaded_rf = pickle.load(f)
            if st.sidebar.button("Predict", key = "predict_rf"):
                y_pred = loaded_rf.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                report = classification_report(
                        y_test, y_pred, labels=labels, output_dict=True, zero_division=0)
                df = pd.DataFrame(report).transpose().round(3)

                st.write(f"Accuracy: **{acc:.3f}**")
                st.table(df.loc[labels, ['precision', 'recall', 'f1-score']])

                if plot_cm_choice == 'Yes':
                    plot_conf_matrix(y_test, y_pred, labels)



if __name__ == '__main__':
    main()