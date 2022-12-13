import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

model = joblib.load("./models/final_model.pkl")

emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}

def predict_emotion(text):
    return model.predict([text])[0]

def predict_emotion_proba(text):
    return model.predict_proba([text])

def main():
    streamlit_style = """
			<style>
			@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400&display=swap');
			html, body, [class*="css"]  {
			font-family: 'Poppins'!important;
			}
			</style>
			"""
    st.markdown(streamlit_style, unsafe_allow_html=True)

    hide_table_row_index = """
                <style>
                thead tr th:first-child {display:none}
                tbody th {display:none}
                </style>
                """
    st.markdown(hide_table_row_index, unsafe_allow_html=True)

    st.title("Emotion Classifier App")
    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Home":
        st.subheader("Home Section")
        
        with st.form(key="emotion_clf_form"):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button("Submit")
        
        if submit_text:
            col1, col2 = st.columns(2)

            prediction = predict_emotion(raw_text)
            prediction_proba = predict_emotion_proba(raw_text)

            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                emoji = emotions_emoji_dict.get(prediction, "")
                st.write(f"{prediction}: {emoji}")

            with col2:
                st.success("Prediction Probability")
                df = pd.DataFrame(prediction_proba, columns=model.classes_)
                df = df.T.reset_index()
                df.columns = ["Emotion", "Probability"]
                st.table(df)
            
            container = st.container()
            container.success("Prediction Probability Graph")
            fig, ax = plt.subplots()
            ax.bar(model.classes_, prediction_proba[0], color="lightgreen", edgecolor="black")
            ax.set_xlabel("Emotion")
            ax.set_ylabel("Probability")
            container.pyplot(fig)
    
    else:
        st.subheader("About Section")
        st.info("This app was created using Pandas, ScikitLearn and Streamlit. 😎")

if __name__ == "__main__":
    main()