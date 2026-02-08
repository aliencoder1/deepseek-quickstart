'''
Author: xuxinyi
Date: 2026-02-08 16:24:24
LastEditors: xuxinyi
LastEditTime: 2026-02-08 16:25:37
FilePath: /deepseek-quickstart/多模态课程/nlp-homework.py
'''
# encoding=utf-8

"""
pip install pandas scikit-learn
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# 读取数据（由于文件没有表头，所以header=None）
# 使用脚本所在目录的路径
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
train_df = pd.read_csv(os.path.join(script_dir, 'training.csv'), header=None)
test_df = pd.read_csv(os.path.join(script_dir, 'testing.csv'), header=None)

# 提取特征和标签
y = train_df[0]              # 类别标签
X = train_df[1].astype(str)  # 文本内容

# 划分出训练集和验证集(假设80%用于训练,20%用于验证)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 用TF-IDF进行特征工程处理（analyzer='char'表示按字符处理，这样就不需要分词了；ngram_range=(1,3)表示单字、双字、三字的组合）
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), max_features=30000)

# 在训练集上拟合
X_train_tfidf = vectorizer.fit_transform(X_train)

feature_names = vectorizer.get_feature_names_out()
idf_values = vectorizer.idf_
df_tfidf = pd.DataFrame({'特征词': feature_names, '权重': idf_values})
df_tfidf = df_tfidf.sort_values(by='权重', ascending=False)
print(df_tfidf.head(100))

# 转换训练集和验证集（汉字->向量）
X_val_tfidf = vectorizer.transform(X_val)

# 用线性SVC(LinearSVC)训练模型
clf = LinearSVC(random_state=42)
clf.fit(X_train_tfidf, y_train)

# 验证模型效果
val_predictions = clf.predict(X_val_tfidf)
accuracy = accuracy_score(y_val, val_predictions)

# 打印输出准确率
print(f"验证集准确率: {accuracy:.4f}")

# 生成最终测试集的预测结果
X_full_tfidf = vectorizer.fit_transform(X)
clf.fit(X_full_tfidf, y)

# 对测试集进行预测
X_test = test_df[1].astype(str)
X_test_tfidf = vectorizer.transform(X_test)
test_predictions = clf.predict(X_test_tfidf)

# 保存结果到当前目录（prediction_result.csv文件）
submission = pd.DataFrame({'ID': test_df[0], 'Label': test_predictions})
output_path = os.path.join(os.getcwd(), 'prediction_result.csv')
submission.to_csv(output_path, index=False, header=False)
print(f"预测结果已保存到: {output_path}")



