import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 设置 Streamlit 页面配置
st.set_page_config(page_title="Iris Classifier", layout='wide')

# 页面标题
st.title("🌺 Iris Flower Classification App")
st.sidebar.title("Model Parameters")  # 侧边栏标题
st.write("This application predicts the species of an iris flower using a **Random Forest Classifier**.")

# 加载 Iris 数据集
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
species_map = dict(enumerate(iris.target_names))
df['species'] = df['species'].map(species_map)

# 显示数据概览
st.subheader("Iris Dataset Overview")
st.dataframe(df.sample(10), use_container_width=True)  # 随机显示 10 条数据

# **数据可视化**
st.subheader("Data Visualization")

# 分栏布局
col1, col2 = st.columns([2, 1])

# **柱状图：类别分布**
with col1:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(x='species', hue='species', data=df, palette="Set2", legend=False, ax=ax)
    ax.set_title("Class Distribution", fontsize=14, fontweight='bold')
    ax.set_xlabel("Species", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    st.pyplot(fig)

# **散点图：花瓣长度 vs 花萼长度**
with col2:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=df['sepal length (cm)'], y=df['petal length (cm)'], hue=df['species'], palette="Dark2", ax=ax)
    ax.set_title("Sepal vs Petal Length", fontsize=14, fontweight='bold')
    ax.set_xlabel("Sepal Length (cm)", fontsize=12)
    ax.set_ylabel("Petal Length (cm)", fontsize=12)
    st.pyplot(fig)

# **模型训练**
st.subheader("Train and Evaluate the Model")

# 训练测试数据拆分
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建和训练模型
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)

# 显示模型准确率
accuracy = accuracy_score(y_test, y_pred)
st.write(f"**Model Accuracy:** {accuracy:.2%}")

# 预测功能
st.subheader("Make a Prediction")

# 在侧边栏获取用户输入
sepal_length = st.sidebar.slider("Sepal Length (cm)", float(df["sepal length (cm)"].min()), float(df["sepal length (cm)"].max()), float(df["sepal length (cm)"].mean()))
sepal_width = st.sidebar.slider("Sepal Width (cm)", float(df["sepal width (cm)"].min()), float(df["sepal width (cm)"].max()), float(df["sepal width (cm)"].mean()))
petal_length = st.sidebar.slider("Petal Length (cm)", float(df["petal length (cm)"].min()), float(df["petal length (cm)"].max()), float(df["petal length (cm)"].mean()))
petal_width = st.sidebar.slider("Petal Width (cm)", float(df["petal width (cm)"].min()), float(df["petal width (cm)"].max()), float(df["petal width (cm)"].mean()))

# 进行预测
user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = rf_clf.predict(user_input)
predicted_species = species_map[prediction[0]]

st.write(f"### Predicted Species: 🌿 **{predicted_species.capitalize()}**")

# 展示 Streamlit 气球动画 🎈
st.balloons()
