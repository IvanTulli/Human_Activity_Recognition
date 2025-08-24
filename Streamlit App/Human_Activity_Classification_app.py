import time, streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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
    with st.expander("Information"):
        st.markdown('We are using the Kaggle dataset: https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones.')
        st.markdown("The aim is to determine whether a person is laying, "
        "sitting, standing, walking, walking downstairs, or walking upstairs, " \
        "given the gyroscope and accelerometer readings of their smartphone.")
        st.markdown("This app shows the prediction accuracy, recall, precision, and f1-score, either on a model trained by the app, "
        "or by a model that is pre-trained.")

    @st.cache_data
    def load_data():
        BASE = Path(__file__).resolve().parent
        TRAIN_PATH = BASE / 'train.csv'
        TEST_PATH = BASE / 'test.csv'
        df_train = pd.read_csv(TRAIN_PATH)
        df_test  = pd.read_csv(TEST_PATH)

        # drop subject column from features
        df_train = df_train.drop(columns=['subject'])
        X_train = df_train.iloc[:, :-1]
        y_train = df_train.iloc[:, -1]

        X_test  = df_test.drop(columns=['subject']).iloc[:, :-1]
        y_test  = df_test.iloc[:, -1]
        return X_train, X_test, y_train, y_test

    def predict(model, x_test):
        y_pred = model.predict(x_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(
                    y_test, y_pred, labels=labels, output_dict=True, zero_division=0
                    )
        df_report = pd.DataFrame(report).transpose().round(3)
        return acc, df_report, y_pred



    def plot_conf_matrix(y_true, y_pred, label_names):
        cm = confusion_matrix(y_true, y_pred, labels=label_names)
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay(cm, display_labels=label_names).plot(
            ax=ax, colorbar=True, xticks_rotation=90
        )
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix")
        st.pyplot(fig)

    X_train, X_test, y_train, y_test = load_data()
    labels = sorted(y_train.unique().tolist())   
    trainchoice= st.sidebar.selectbox("Make a choice:", ("-", "Train a new model", "Use a trained model"))
    if trainchoice == "Train a new model":
        clf_choice = st.sidebar.selectbox(
            'Classifier',
            ('-','Random Choice (Baseline)', 'Support Vector Machine', 'Logistic Regression', 'Random Forest')
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

        if clf_choice == 'Support Vector Machine':
            st.sidebar.subheader("Model Hyperparameters")
            C = st.sidebar.number_input("C (Regularization)", 0.01, 10.0, value=1.0, step=0.01, key='C_SVM')
            kernel = st.sidebar.radio('Kernel', ('linear', 'rbf', 'poly'))
            if st.sidebar.button("Classify", key='classify_svm'):
                st.subheader("Support Vector Machine Results")

                with st.spinner("Training model..."):
                
                    model = make_pipeline(
                        StandardScaler(),
                        SVC(C=C, kernel=kernel)
                    )
                    model.fit(X_train, y_train)
                    time.sleep(0.5)  
                    st.success("Training complete!")
                with st.spinner("Making predictions..."):
                    time.sleep(0.5)

                    acc, df_report, y_pred = predict(model, X_test)

                    st.write(f"Accuracy: **{acc:.3f}**")
                    st.table(df_report.loc[labels, ['precision', 'recall', 'f1-score']])

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
                
                with st.spinner("Making predictions..."):
                    time.sleep(0.5)
                    acc, df_report, y_pred = predict(model, X_test)

                    st.write(f"Accuracy: **{acc:.3f}**")
                    st.table(df_report.loc[labels, ['precision', 'recall', 'f1-score']])

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
                
                with st.spinner("Making predictions..."):
                    time.sleep(0.5)
                    acc, df_report, y_pred = predict(rf, X_test)

                    st.write(f"Accuracy: **{acc:.3f}**")
                    st.table(df_report.loc[labels, ['precision', 'recall', 'f1-score']])

                    if plot_cm_choice == 'Yes':
                        plot_conf_matrix(y_test, y_pred, labels)

    elif trainchoice == "Use a trained model":
        clf_choice = st.sidebar.selectbox('Choose trained classifier', ('-', 'Support Vector Machine', 'Logistic Regression', 'Random Forest'))
        plot_cm_choice = st.sidebar.radio("Plot the confusion matrix?", ('Yes', 'No'))
        if clf_choice == "Support Vector Machine":
            with open("svm_model.pkl", "rb") as f:
                loaded_svm = pickle.load(f)
            if st.sidebar.button("Predict", key = "predict_svm"):
                st.subheader('Support Vector Machine Predictions')
                y_pred = loaded_svm.predict(X_test)

                acc, df_report, y_pred = predict(loaded_svm, X_test)

                st.write(f"Accuracy: **{acc:.3f}**")
                st.table(df_report.loc[labels, ['precision', 'recall', 'f1-score']])

                if plot_cm_choice == 'Yes':
                    plot_conf_matrix(y_test, y_pred, labels)
        if clf_choice == "Logistic Regression":
            with open("lr_model.pkl", "rb") as f:
                loaded_lr = pickle.load(f)
            if st.sidebar.button("Predict", key = "predict_lr"):
                st.subheader("Logistic Regressions Predictions")
                y_pred = loaded_lr.predict(X_test)

                acc, df_report, y_pred = predict(loaded_lr, X_test)

                st.write(f"Accuracy: **{acc:.3f}**")
                st.table(df_report.loc[labels, ['precision', 'recall', 'f1-score']])

                if plot_cm_choice == 'Yes':
                    plot_conf_matrix(y_test, y_pred, labels)

        if clf_choice == "Random Forest":
            with open("rf_model.pkl", "rb") as f:
                loaded_rf = pickle.load(f)
            if st.sidebar.button("Predict", key = "predict_rf"):
                st.subheader("Random Forest Predictions")
                y_pred = loaded_rf.predict(X_test)

                acc, df_report, y_pred = predict(loaded_rf, X_test)

                st.write(f"Accuracy: **{acc:.3f}**")
                st.table(df_report.loc[labels, ['precision', 'recall', 'f1-score']])

                if plot_cm_choice == 'Yes':
                    plot_conf_matrix(y_test, y_pred, labels)



if __name__ == '__main__':
    main()