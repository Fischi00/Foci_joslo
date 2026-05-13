import streamlit as st
import pickle
import os
import warnings

# Sklearn warningok elrejtése
warnings.filterwarnings("ignore", category=UserWarning)

# --- OLDAL BEÁLLÍTÁSAI ---
st.set_page_config(page_title="Foci Jósló MI", page_icon="⚽", layout="wide")

# --- MODELLEK BETÖLTÉSE (Gyorsítótárazva!) ---
# Az @st.cache_resource gondoskodik róla, hogy a modellek csak egyszer töltődjenek be a memóriába, ne minden kattintásnál.
@st.cache_resource
def load_resources():
    # Kiszámoljuk a models mappa helyét
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, 'Backend', 'models')
    
    team_le = pickle.load(open(os.path.join(MODEL_PATH, "team_encoder.pkl"), "rb"))
    team_stats = pickle.load(open(os.path.join(MODEL_PATH, "team_stats.pkl"), "rb"))
    scaler = pickle.load(open(os.path.join(MODEL_PATH, "scaler.pkl"), "rb"))
    teams_by_league = pickle.load(open(os.path.join(MODEL_PATH, "teams_by_league.pkl"), "rb"))
    
    models = {
        "XGBoost": pickle.load(open(os.path.join(MODEL_PATH, "XGBoost_model.pkl"), "rb")),
        "RandomForest": pickle.load(open(os.path.join(MODEL_PATH, "RandomForest_model.pkl"), "rb")),
        "Logistic": pickle.load(open(os.path.join(MODEL_PATH, "Logistic_model.pkl"), "rb")),
        "NeuralNet": pickle.load(open(os.path.join(MODEL_PATH, "NeuralNet_model.pkl"), "rb"))
    }
    return team_le, team_stats, scaler, teams_by_league, models

try:
    team_le, team_stats, scaler, teams_by_league, models = load_resources()
except Exception as e:
    st.error(f"Hiba a modellek betöltésekor: {e}. Biztosan lefutott a train.py?")
    st.stop()

# --- FELHASZNÁLÓI FELÜLET (UI) ---
st.title("⚽ Foci Jósló MI: Modell Párbaj")
st.markdown("Válaszd ki a bajnokságokat, a csapatokat és a modelleket, hogy kiderítsd, ki nyeri a meccset!")

leagues = list(teams_by_league.keys())

# Hazai és Vendég oszlopok létrehozása
col1, col2 = st.columns(2)

with col1:
    st.header("🏠 Hazai Oldal")
    home_league = st.selectbox("Bajnokság (Hazai)", leagues, key="hl")
    home_team = st.selectbox("Csapat (Hazai)", teams_by_league[home_league], key="ht")

with col2:
    st.header("🚀 Vendég Oldal")
    away_league = st.selectbox("Bajnokság (Vendég)", leagues, key="al")
    away_team = st.selectbox("Csapat (Vendég)", teams_by_league[away_league], key="at")

st.markdown("---")
st.subheader("🤖 Modellek Kiválasztása")
model_names = list(models.keys())

col3, col4 = st.columns(2)
with col3:
    model1_name = st.selectbox("1. Modell (Piros sarok)", model_names, index=0) # Alapból XGBoost
with col4:
    model2_name = st.selectbox("2. Modell (Kék sarok)", model_names, index=3) # Alapból NeuralNet

# --- GOMB ÉS JÓSLÁS LOGIKA ---
# A st.button csak akkor ad vissza True-t, ha rákattintanak
if st.button("Jóslat Futtatása 🚀", use_container_width=True, type="primary"):
    with st.spinner('A mesterséges intelligencia elemzi a formákat és a statisztikákat...'):
        
        # 1. Adatok kinyerése
        h_id = team_le.transform([home_team])[0]
        a_id = team_le.transform([away_team])[0]
        
        home_data = team_stats.get(home_team, {'HS': 10, 'HST': 4, 'Home_Form': 7})
        away_data = team_stats.get(away_team, {'AS': 10, 'AST': 4, 'Away_Form': 7})
        
        raw_hs = home_data.get('HS', 10)
        raw_hst = home_data.get('HST', 4)
        raw_h_form = home_data.get('Home_Form', 7)
        
        raw_as = away_data.get('AS', 10)
        raw_ast = away_data.get('AST', 4)
        raw_a_form = away_data.get('Away_Form', 7)
        
        # 2. Normalizálás
        stats_array = [[raw_hs, raw_as, raw_hst, raw_ast, raw_h_form, raw_a_form]]
        scaled_stats = scaler.transform(stats_array)[0]
        
        features = [[h_id, a_id, scaled_stats[0], scaled_stats[1], scaled_stats[2], scaled_stats[3], scaled_stats[4], scaled_stats[5]]]
        
        # 3. Jóslás
        res_map = {0: "Vendég (Away) győzelem 🔴", 1: "Döntetlen (Draw) ⚪", 2: "Hazai (Home) győzelem 🟢"}
        
        pred1 = models[model1_name].predict(features)[0]
        pred2 = models[model2_name].predict(features)[0]
        
        result1 = res_map[int(pred1)]
        result2 = res_map[int(pred2)]
        
        # --- EREDMÉNYEK KIÍRÁSA ---
        st.markdown("---")
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.error(f"**{model1_name}** jóslata:\n### {result1}")
            
        with res_col2:
            st.info(f"**{model2_name}** jóslata:\n### {result2}")