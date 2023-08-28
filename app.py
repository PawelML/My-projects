import streamlit as st
import openai
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
#import matplotlib

#matplotlib.use('TkAgg')


openai.api_key = 'your_api_key'

llm = OpenAI(api_token=openai.api_key )
pandas_ai = PandasAI(llm)


st.title('Prompt-driven analysis with PandasAI')
uploaded_file = st.file_uploader('Upload a CSV file for analysis', type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head(3))

    prompt = st.text_area('Enter your prompt:')

    if st.button('Generate'):
        if prompt:
            with st.spinner('Generating response...'):
                st.write(pandas_ai.run(df, prompt=prompt))
        else:
            st.warning('Please enter a prompt.')