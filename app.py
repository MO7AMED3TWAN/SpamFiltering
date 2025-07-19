import streamlit as st


st.title("📧 Spam Mail Filter")
st.write("Classify emails as SPAM or HAM using your custom model")

with st.sidebar:
    st.header("Model Training")
    uploaded_file = st.file_uploader("Upload training CSV", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        if st.button("Train Model"):
            with st.spinner("Training..."):
                X = data["text"]
                y = data["label"]
                pipeline.fit(X, y)
                joblib.dump(pipeline, 'spam_model.joblib')
                st.success("Model trained!")

tab1, tab2 = st.tabs(["Single Email", "Batch Process"])

with tab1:
    email = st.text_area("Paste email text:", height=200)
    if st.button("Classify Email"):
        try:
            model = joblib.load('spam_model.joblib')
            pred = model.predict([email])[0]
            proba = model.predict_proba([email])[0]
            if pred == 1:
                st.error(f"🚨 SPAM ({proba[1]*100:.1f}% confidence)")
            else:
                st.success(f"✅ HAM ({proba[0]*100:.1f}% confidence)")
        except:
            st.warning("Train a model first")

with tab2:
    batch_file = st.file_uploader("Upload emails CSV", type=["csv"], key="batch")
    if batch_file:
        batch_data = pd.read_csv(batch_file)
        if st.button("Process Batch"):
            try:
                model = joblib.load('spam_model.joblib')
                batch_data['prediction'] = model.predict(batch_data["text"])
                st.dataframe(batch_data)
                
                csv = batch_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Results",
                    data=csv,
                    file_name='predictions.csv'
                )
            except:
                st.warning("Train a model first")