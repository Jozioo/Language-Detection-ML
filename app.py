import streamlit as sl
import joblib

model_mnb = joblib.load("saved_models/mnb_language_detector.pkl")
model_lr = joblib.load("saved_models/logistic_regression_pipeline.pkl")
model_svm = joblib.load("saved_models/linear_svm_pipeline.pkl")
model_fs = joblib.load("saved_models/fasttext_style_pipeline.pkl")
model_knn = joblib.load("saved_models/knn_pipeline.pkl")

def predict_mnb(text):
    lang = model_mnb.predict([text])[0]
    proba = model_mnb.predict_proba([text])[0]
    return lang, float(proba.max())

def predict_lr(text):
    lang = model_lr.predict([text])[0]
    proba = model_lr.predict_proba([text])[0]
    return lang, float(proba.max())

def predict_svm(text):
    lang = model_svm.predict([text])[0]
    proba = model_svm.predict_proba([text])[0]
    return lang, float(proba.max())

def predict_fs(text):
    lang = model_fs.predict([text])[0]
    proba = model_fs.predict_proba([text])[0]
    return lang, float(proba.max())

def predict_knn(text):
    lang = model_knn.predict([text])[0]
    proba = model_knn.predict_proba([text])[0]
    return lang, float(proba.max())

sl.title("Language Detector")

choice = sl.selectbox(
    "Pick model",
    ["MNB", "Logistic Regression", "SVM", "FastText", "KNN", "All Models"]
)

text = sl.text_input("Input your text")

if text:
    sl.write("Our prediction & Confidence")

    if choice == "MNB":
        lang, conf = predict_mnb(text)
        sl.write(f"MNB: {lang} / Confidence ({conf*100:.1f}%)")
        sl.progress(conf)

    elif choice == "Logistic Regression":
        lang, conf = predict_lr(text)
        sl.write(f"LR: {lang} / Confidence ({conf*100:.1f}%)")
        sl.progress(conf)

    elif choice == "SVM":
        lang, conf = predict_svm(text)
        sl.write(f"SVM: {lang} / Confidence ({conf*100:.1f}%)")
        sl.progress(conf)

    elif choice == "FastText":
        lang, conf = predict_fs(text)
        sl.write(f"FastText: {lang} / Confidence ({conf*100:.1f}%)")
        sl.progress(conf)

    elif choice == "KNN":
        lang, conf = predict_knn(text)
        sl.write(f"KNN: {lang} / Confidence ({conf*100:.1f}%)")
        sl.progress(conf)

    elif choice == "All Models":
        lang_mnb, conf_mnb = predict_mnb(text)
        lang_lr,  conf_lr  = predict_lr(text)
        lang_svm, conf_svm = predict_svm(text)
        lang_fs,  conf_fs  = predict_fs(text)
        lang_knn, conf_knn = predict_knn(text)

        sl.write(f"MNB      : {lang_mnb} / Confidence ({conf_mnb*100:.1f}%)")
        sl.progress(conf_mnb)
        sl.write(f"LR       : {lang_lr}  / Confidence ({conf_lr*100:.1f}%)")
        sl.progress(conf_lr)
        sl.write(f"SVM      : {lang_svm} / Confidence ({conf_svm*100:.1f}%)")
        sl.progress(conf_svm)
        sl.write(f"FastText : {lang_fs}  / Confidence ({conf_fs*100:.1f}%)")
        sl.progress(conf_fs)
        sl.write(f"KNN      : {lang_knn} / Confidence ({conf_knn*100:.1f}%)")
        sl.progress(conf_knn)