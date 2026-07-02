import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# --- АВТОМАТИЧЕСКАЯ НАСТРОЙКА ПУТЕЙ ---
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, '..', 'data', 'raw', 'netflix_titles.csv')
output_dir = os.path.join(current_dir, '..', 'img', 'img')
os.makedirs(output_dir, exist_ok=True)

# --- ЗАГРУЗКА ДАННЫХ ---
try:
    df = pd.read_csv(csv_path)
    print(f"[ОК] Датасет загружен. Строк: {df.shape[0]}")
except FileNotFoundError:
    # Запасной абсолютный путь, если относительный не сработал
    csv_path = 'C:/Users/eaman/Downloads/files/netflix-project/netflix-project/data/raw/netflix_titles.csv'
    df = pd.read_csv(csv_path)

# --- ЭТАП 1: МАТЕМАТИКО-СТАТИСТИЧЕСКИЙ АНАЛИЗ ---
stats = df['release_year'].describe()

# Выбросы: Boxplot
plt.figure(figsize=(8, 4))
sns.boxplot(x=df['release_year'], color='#E50914')
plt.title('Анализ выбросов: Год выпуска контента (Boxplot)', fontsize=12, fontweight='bold')
plt.savefig(os.path.join(output_dir, 'boxplot_release_year.png'), bbox_inches='tight', dpi=300)
plt.close()

# Распределение целевой переменной
plt.figure(figsize=(6, 4))
sns.countplot(x='type', hue='type', data=df, palette=['#E50914', '#222222'], legend=False)
plt.title('Распределение целевой переменной (Type)', fontsize=12, fontweight='bold')
plt.savefig(os.path.join(output_dir, 'target_distribution.png'), bbox_inches='tight', dpi=300)
plt.close()

# --- ЭТАП 2: ОБУЧЕНИЕ ML-МОДЕЛИ ---
df_ml = df[['type', 'description']].dropna()
X = df_ml['description']
y = df_ml['type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Важно: class_weight='balanced' спасает от нулевой точности сериалов!
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)

# Матрица ошибок
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Матрица ошибок (Confusion Matrix)', fontsize=12, fontweight='bold')
plt.ylabel('Реальные классы')
plt.xlabel('Предсказанные моделью')
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), bbox_inches='tight', dpi=300)
plt.close()

# Важность признаков
importances = model.feature_importances_
indices = np.argsort(importances)[-10:]
words = np.array(tfidf.get_feature_names_out())[indices]

plt.figure(figsize=(10, 5))
plt.barh(range(len(indices)), importances[indices], color='#E50914')
plt.yticks(range(len(indices)), words, fontsize=11)
plt.title('Топ-10 самых важных признаков/слов в описании', fontsize=12, fontweight='bold')
plt.savefig(os.path.join(output_dir, 'feature_importance.png'), bbox_inches='tight', dpi=300)
print("[УСПЕХ] Все графики обновлены в img/img/!")