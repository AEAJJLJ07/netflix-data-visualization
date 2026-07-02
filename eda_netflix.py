import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# --- НАСТРОЙКА ПУТЕЙ ---
# Указываем точный путь к твоему файлу в папке data/raw/
csv_path = 'C:/Users/eaman/Downloads/files/netflix-project/netflix-project/data/raw/netflix_titles.csv'
output_dir = '../img/img/'
os.makedirs(output_dir, exist_ok=True)

# --- ЗАГРУЗКА ДАННЫХ ---
try:
    df = pd.read_csv(csv_path)
    print(f"[ОК] Датасет успешно загружен! Строк: {df.shape[0]}")
except FileNotFoundError:
    print(f"[ОШИБКА] Не найден файл по пути: {csv_path}")
    print("Проверь, чтобы файл назывался именно netflix_titles.csv (с расширением .csv)")
    exit()

# Настройка стилей для графиков
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ==============================================================================
# ЭТАП 1-2: ВИЗУАЛИЗАЦИЯ И АНАЛИЗ (Недели 1-2)
# ==============================================================================
print("\n--- ГЕНЕРАЦИЯ ГРАФИКОВ ---")

# 1. График пропущенных значений
missing_data = df.isnull().mean() * 100
missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
if not missing_data.empty:
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x=missing_data.values, y=missing_data.index, color='#E50914')
    plt.title('Пропущенные значения по столбцам — Netflix Dataset', fontsize=14, fontweight='bold')
    plt.xlim(0, 30)
    for i, v in enumerate(missing_data.values):
        ax.text(v + 0.5, i, f"{v:.2f}%", va='center', fontweight='bold')
    plt.savefig(os.path.join(output_dir, 'missing_values.png'), bbox_inches='tight', dpi=300)
    plt.close()
print("[+] График пропусков сохранен в папки img/img/missing_values.png")

# 2. Круговая диаграмма типов контента
if 'type' in df.columns:
    plt.figure(figsize=(8, 6))
    type_counts = df['type'].value_counts()
    plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', 
            colors=['#E50914', '#222222'], startangle=90, 
            explode=(0.05, 0), textprops={'fontsize': 12, 'fontweight': 'bold'})
    plt.title('Доля фильмов и сериалов на платформе', fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(output_dir, 'type_distribution.png'), bbox_inches='tight', dpi=300)
    plt.close()
print("[+] Круговая диаграмма типов сохранена в img/img/type_distribution.png")

# 3. Топ-10 жанров
if 'listed_in' in df.columns:
    plt.figure(figsize=(12, 6))
    genres_series = df['listed_in'].str.split(', ').explode()
    top_genres = genres_series.value_counts().head(10)
    sns.barplot(x=top_genres.values, y=top_genres.index, palette='Reds_r')
    plt.title('Топ-10 популярных жанров на Netflix', fontsize=14, fontweight='bold')
    for i, v in enumerate(top_genres.values):
        plt.text(v + 10, i, str(v), va='center', fontweight='bold')
    plt.savefig(os.path.join(output_dir, 'top_genres_horizontal.png'), bbox_inches='tight', dpi=300)
    plt.close()
print("[+] График топ-10 жанров сохранен в img/img/top_genres_horizontal.png")


# ==============================================================================
# ЭТАП 3: МАШИННОЕ ОБУЧЕНИЕ (Неделя 3)
# ==============================================================================
print("\n--- ЭТАП 3: ОБУЧЕНИЕ ML-МОДЕЛИ ---")

# Отбираем нужные колонки и удаляем строки с пропусками в них
df_ml = df[['type', 'description', 'listed_in']].dropna()

# Объединяем жанры и описание в единый текст для обучения модели
df_ml['text_features'] = df_ml['listed_in'] + " " + df_ml['description']

X = df_ml['text_features']
y = df_ml['type']

# Делим выборку на обучение (80%) и тест (20%) с сохранением пропорций классов
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Размер обучающей выборки: {X_train.shape[0]} строк")
print(f"Размер тестовой выборки: {X_test.shape[0]} строк")

# Превращаем текст в набор чисел (Векторизация TF-IDF)
print("Преобразование текстовых признаков в векторы (TF-IDF)...")
tfidf = TfidfVectorizer(max_features=3000, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Создаем и обучаем модель Случайного Леса (Random Forest)
print("Обучение модели Random Forest Classifier (может занять пару секунд)...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_tfidf, y_train)

# Делаем предсказание на тестовых данных
y_pred = model.predict(X_test_tfidf)

# Считаем и выводим метрики эффективности
accuracy = accuracy_score(y_test, y_pred)
print("\n================ МЕТРИКИ ML МОДЕЛИ ================")
print(f"Общая точность модели (Accuracy): {accuracy * 100:.2f}%")
print("\nДетальный отчет по классам (Classification Report):")
print(classification_report(y_test, y_pred))
print("====================================================")

print("\n[УСПЕХ] Вся программа (Недели 1-3) успешно выполнена!")