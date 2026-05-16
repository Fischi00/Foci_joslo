import streamlit as st
import pickle
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(page_title="Foci Jósló MI", page_icon="⚽", layout="wide")

@st.cache_resource
def load_resources():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, 'Backend', 'models')
    
    team_le = pickle.load(open(os.path.join(MODEL_PATH, "team_encoder.pkl"), "rb"))
    team_stats = pickle.load(open(os.path.join(MODEL_PATH, "team_stats.pkl"), "rb"))
    scaler = pickle.load(open(os.path.join(MODEL_PATH, "scaler.pkl"), "rb"))
    teams_by_league = pickle.load(open(os.path.join(MODEL_PATH, "teams_by_league.pkl"), "rb"))
    h2h_matches = pickle.load(open(os.path.join(MODEL_PATH, "h2h_matches.pkl"), "rb")) # ÚJ!
    
    models = {
        "XGBoost": pickle.load(open(os.path.join(MODEL_PATH, "XGBoost_model.pkl"), "rb")),
        "RandomForest": pickle.load(open(os.path.join(MODEL_PATH, "RandomForest_model.pkl"), "rb")),
        "Logistic": pickle.load(open(os.path.join(MODEL_PATH, "Logistic_model.pkl"), "rb")),
        "NeuralNet": pickle.load(open(os.path.join(MODEL_PATH, "NeuralNet_model.pkl"), "rb"))
    }
    return team_le, team_stats, scaler, teams_by_league, h2h_matches, models

try:
    team_le, team_stats, scaler, teams_by_league, h2h_matches, models = load_resources()
except Exception as e:
    st.error(f"Hiba a modellek betöltésekor: {e}")
    st.stop()

st.title("⚽ Foci Jósló MI: Modell Párbaj")
st.markdown("Válaszd ki a bajnokságokat, a csapatokat és a modelleket, majd ellenőrizd a modellek pontosságát a valósággal!")

leagues = list(teams_by_league.keys())

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
    model1_name = st.selectbox("1. Modell (Piros sarok)", model_names, index=0)
with col4:
    model2_name = st.selectbox("2. Modell (Kék sarok)", model_names, index=3)

if st.button("Jóslat Futtatása és Validáció 🚀", use_container_width=True, type="primary"):
    with st.spinner('Elemzés folyamatban...'):
        h_id = team_le.transform([home_team])[0]
        a_id = team_le.transform([away_team])[0]
        
        home_data = team_stats.get(home_team, {'HS': 10, 'HST': 4, 'Home_Form': 7})
        away_data = team_stats.get(away_team, {'AS': 10, 'AST': 4, 'Away_Form': 7})
        
        raw_hs, raw_hst, raw_h_form = home_data.get('HS', 10), home_data.get('HST', 4), home_data.get('Home_Form', 7)
        raw_as, raw_ast, raw_a_form = away_data.get('AS', 10), away_data.get('AST', 4), away_data.get('Away_Form', 7)
        
        stats_array = [[raw_hs, raw_as, raw_hst, raw_ast, raw_h_form, raw_a_form]]
        scaled_stats = scaler.transform(stats_array)[0]
        features = [[h_id, a_id, scaled_stats[0], scaled_stats[1], scaled_stats[2], scaled_stats[3], scaled_stats[4], scaled_stats[5]]]
        
        res_map = {0: "Vendég győzelem 🔴", 1: "Döntetlen ⚪", 2: "Hazai győzelem 🟢"}
        pred_to_ftr = {0: "A", 1: "D", 2: "H"} # Git-hez és összehasonlításhoz
        
        pred1 = models[model1_name].predict(features)[0]
        pred2 = models[model2_name].predict(features)[0]
        
        # --- MODELLEK TIPPJEINEK KIÍRÁSA ---
        st.markdown("---")
        st.subheader("🔮 A Modellek Tippjei")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.error(f"**{model1_name}** jóslata:\n### {res_map[int(pred1)]}")
        with res_col2:
            st.info(f"**{model2_name}** jóslata:\n### {res_map[int(pred2)]}")
            
        # --- ÚJ RÉSZ: VALÓSÁG ELLENŐRZÉSE (VALIDÁCIÓ) ---
        st.markdown("---")
        st.subheader("📊 Valóság Ellenőrzése (Történelmi Visszatekintés)")
        
        h2h_key = f"{home_team} vs {away_team}"
        if h2h_key in h2h_matches:
            actual = h2h_matches[h2h_key]
            actual_ftr = actual['FTR']
            
            actual_map = {"H": "Hazai győzelem 🟢", "D": "Döntetlen ⚪", "A": "Vendég győzelem 🔴"}
            
            st.markdown(f"A két csapat **legutóbbi egymás elleni mérkőzése** az adatbázisban (Dátum: {actual['Date']}):")
            st.success(f"### {home_team}  {actual['Score']}  {away_team}  ➔  Végeredmény: {actual_map[actual_ftr]}")
            
            # Pipák és X-ek kirakása
            val_col1, val_col2 = st.columns(2)
            with val_col1:
                if pred_to_ftr[int(pred1)] == actual_ftr:
                    st.success(f"🎯 **{model1_name}** Eltalálta! (Sikeres predikció)")
                else:
                    st.error(f"❌ **{model1_name}** Tévedett.")
            with val_col2:
                if pred_to_ftr[int(pred2)] == actual_ftr:
                    st.success(f"🎯 **{model2_name}** Eltalálta! (Sikeres predikció)")
                else:
                    st.error(f"❌ **{model2_name}** Tévedett.")
        else:
            st.warning("⚠️ Erre a konkrét Hazai-Vendég felállásra nincs közvetlen korábbi meccs az adatbázisban (pl. nem azonos ligában játszanak), így a történelmi validáció nem lehetséges.")