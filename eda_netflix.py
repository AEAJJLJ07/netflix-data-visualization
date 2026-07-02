import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# --- НАСТРОЙКА ПУТЕЙ ---
csv_path = 'C:/Users/eaman/Downloads/files/netflix-project/netflix-project/data/raw/netflix_titles.csv'
output_dir = 'C:/Users/eaman/Downloads/files/netflix-project/netflix-project/img/img/'
os.makedirs(output_dir, exist_ok=True)

# --- ЗАГРУЗКА ДАННЫХ ---
df = pd.read_csv(csv_path)
print(f"[ОК] Датасет загружен. Строк: {df.shape[0]}")

# ==============================================================================
# 8.3 МАТЕМАТИКО-СТАТИСТИЧЕСКИЙ АНАЛИЗ
# ==============================================================================
print("\n--- ЭТАП 1: МАТЕМАТИКО-СТАТИСТИЧЕСКИЙ АНАЛИЗ ---")

# Описательная статистика для числового признака (release_year)
stats = df['release_year'].describe()
print("\n[Описательная статистика для release_year]:")
print(stats)

# Выбросы: Строим Boxplot для release_year
plt.figure(figsize=(8, 4))
sns.boxplot(x=df['release_year'], color='#E50914')
plt.title('Анализ выбросов: Год выпуска контента (Boxplot)', fontsize=12, fontweight='bold')
plt.savefig(os.path.join(output_dir, 'boxplot_release_year.png'), bbox_inches='tight', dpi=300)
plt.close()
print("[+] График выбросов boxplot_release_year.png сохранен.")

# Распределение целевой переменной (type)
plt.figure(figsize=(6, 4))
sns.countplot(x='type', data=df, palette=['#E50914', '#222222'])
plt.title('Распределение целевой переменной (Type)', fontsize=12, fontweight='bold')
plt.savefig(os.path.join(output_dir, 'target_distribution.png'), bbox_inches='tight', dpi=300)
plt.close()

# ==============================================================================
# 8.5 и 8.7 ОБУЧЕНИЕ И ВИЗУАЛИЗАЦИЯ ML-МОДЕЛИ
# ==============================================================================
print("\n--- ЭТАП 2: ЧЕСТНОЕ ОБУЧЕНИЕ ML-МОДЕЛИ ---")

# Чтобы избежать Data Leakage (100% точности), обучаемся ТОЛЬКО на описании (description)
df_ml = df[['type', 'description']].dropna()

X = df_ml['description'] # Признак: текст описания
y = df_ml['type']        # Целевая переменная: Movie или TV Show

# Разделение данных 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Векторизация TF-IDF
tfidf = TfidfVectorizer(max_features=2000, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Обучение Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_tfidf, y_train)

# Предсказание
y_pred = model.predict(X_test_tfidf)

# Метрики качества
accuracy = accuracy_score(y_test, y_pred)
print("\n================ МЕТРИКИ ML МОДЕЛИ ================")
print(f"Общая точность модели (Accuracy): {accuracy * 100:.2f}%")
print("\nДетальный отчет (Classification Report):")
print(classification_report(y_test, y_pred))
print("====================================================")

# Визуализация результатов: Матрица ошибок (Confusion Matrix)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Матрица ошибок (Confusion Matrix)', fontsize=12, fontweight='bold')
plt.ylabel('Реальные классы')
plt.xlabel('Предсказанные моделью')
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), bbox_inches='tight', dpi=300)
plt.close()
print("[+] Матрица ошибок confusion_matrix.png сохранена.")

# Важность признаков (Feature Importance - топ слов)
importances = model.feature_importances_
indices = np.argsort(importances)[-10:]
words = np.array(tfidf.get_feature_names_out())[indices]

plt.figure(figsize=(10, 5))
plt.barh(range(len(indices)), importances[indices], color='#E50914')
plt.yticks(range(len(indices)), words, fontsize=11)
plt.title('Топ-10 самых важных слов для определения типа контента', fontsize=12, fontweight='bold')
plt.savefig(os.path.join(output_dir, 'feature_importance.png'), bbox_inches='tight', dpi=300)
plt.close()
print("[+] График важности признаков feature_importance.png сохранен.")

print("\n[УСПЕХ] Скрипт успешно выполнил все математические и ML требования преподавателя!")
