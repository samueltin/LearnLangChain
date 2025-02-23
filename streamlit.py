import streamlit as st
import pandas as pd
import numpy as np
import QASql2 as query
import ast


st.title("Use Natual Language to Query")

question = st.chat_input("Your Question")

if question:
    df = pd.DataFrame(
        np.random.randn(10, 5), columns=("col %d" % i for i in range(5))
    )
    state = query.write_query_and_execute_query({"question": question})
    result = state["result"]
    
    result_dict = ast.literal_eval(result)
    df = pd.DataFrame(result_dict)  

    st.table(df)