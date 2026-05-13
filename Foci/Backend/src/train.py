import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from data_manager import load_all_football_data
from preprocessor import FootballPreprocessor
from models import get_model_zoo
from config import FEATURES, MODEL_PATH

# 1. Adatok betöltése
print("Adatok beolvasása folyamatban...")
df = load_all_football_data()

# 2. Előfeldolgozás és Normalizálás
print("Adatok előkészítése, statisztikák számítása és normalizálása...")
prep = FootballPreprocessor()
prep.fit_encoders(df)
df = prep.transform_data(df, is_training=True)

# KORRELÁCIÓ VIZSGÁLAT (Mik befolyásolják leginkább a végeredményt?)
print("\n--- KORRELÁCIÓ VIZSGÁLAT ---")
correlation_matrix = df[['Home_Id', 'Away_Id', 'HS', 'AS', 'HST', 'AST', 'Target']].corr()
print(correlation_matrix['Target'].sort_values(ascending=False))
print("----------------------------\n")

# 3. Adatok felosztása (80% tanítás, 20% tesztelés - TELJESÍTMÉNY ANALÍZIS)
X = df[FEATURES]
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# 4. Modellek tanítása és TESZTELÉSE
models = get_model_zoo()
print("--- MODELLEK ÉRTÉKELÉSE ---")
for name, model in models.items():
    # Tanítás a 80%-on
    model.fit(X_train, y_train)
    
    # Vizsgáztatás a 20%-on
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"\n{name} Pontossága (Accuracy): {accuracy * 100:.2f}%")
    print(classification_report(y_test, predictions, target_names=['Away', 'Draw', 'Home']))
    
    # Modell mentése
    save_path = os.path.join(MODEL_PATH, f'{name}_model.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)

print(f"\nMinden modell sikeresen elmentve a {MODEL_PATH} mappába!")