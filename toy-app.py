import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
apptitle = 'DSSI Toy App'
st.set_page_config(page_title=apptitle, layout='wide')

# Page title
st.title('My First Streamlit Application')
st.sidebar.title("Diabetes Data Exploration")  # 添加侧边栏标题
st.write('Reference: [Streamlit API](https://docs.streamlit.io/en/stable/api.html#display-data)')
st.balloons()

# Load diabetes dataset
st.subheader('**Diabetes Data**')
db = datasets.load_diabetes()
df = pd.DataFrame(db.data, columns=db.feature_names)

# Two-column layout
col1, col2 = st.columns([2, 1])

# Display dataframe in col1
with col1:
    st.write("### Diabetes Dataset Overview")
    st.dataframe(df, use_container_width=True)

# Histogram visualization in col2
with col2:
    st.write("### Age Distribution of Patients")

    fig, ax = plt.subplots(figsize=(8, 4))  # 增加图表尺寸
    df['age'].hist(bins=15, color='skyblue', edgecolor='black', alpha=0.75, ax=ax)  # 设置颜色、边界和透明度
    ax.set_title("Age Distribution of Diabetes Patients", fontsize=14, fontweight='bold')  # 设置标题
    ax.set_xlabel("Age (Standardized)", fontsize=12)  # X轴标签
    ax.set_ylabel("Frequency", fontsize=12)  # Y轴标签
    ax.grid(axis='y', linestyle='--', alpha=0.7)  # 添加网格线，提高可读性

    st.pyplot(fig)  # 显示图表
