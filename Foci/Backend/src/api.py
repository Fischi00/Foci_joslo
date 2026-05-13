from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models')

try:
    team_le = pickle.load(open(os.path.join(MODEL_PATH, "team_encoder.pkl"), "rb"))
    team_stats = pickle.load(open(os.path.join(MODEL_PATH, "team_stats.pkl"), "rb"))
    scaler = pickle.load(open(os.path.join(MODEL_PATH, "scaler.pkl"), "rb"))
    
    models = {
        "XGBoost": pickle.load(open(os.path.join(MODEL_PATH, "XGBoost_model.pkl"), "rb")),
        "RandomForest": pickle.load(open(os.path.join(MODEL_PATH, "RandomForest_model.pkl"), "rb")),
        "Logistic": pickle.load(open(os.path.join(MODEL_PATH, "Logistic_model.pkl"), "rb")),
        "NeuralNet": pickle.load(open(os.path.join(MODEL_PATH, "NeuralNet_model.pkl"), "rb"))
    }
    print("Minden modul sikeresen betöltve!")
except Exception as e:
    print(f"Kritikus hiba a betöltéskor: {e}")

# ÚJ VÉGPONT A LIGÁK LEKÉRÉSÉHEZ
@app.route('/teams', methods=['GET'])
def get_teams():
    try:
        teams_by_league = pickle.load(open(os.path.join(MODEL_PATH, "teams_by_league.pkl"), "rb"))
        return jsonify(teams_by_league)
    except Exception as e:
        return jsonify({"Hiba": []})

@app.route('/predict', methods=['POST'])
def predict_match():
    data = request.json
    home_team = data.get('home')
    away_team = data.get('away')
    selected_model_name = data.get('model', 'XGBoost')
    
    try:
        h_id = team_le.transform([home_team])[0]
        a_id = team_le.transform([away_team])[0]
        
        home_data = team_stats.get(home_team, {'HS': 10, 'HST': 4, 'Home_Form': 7})
        away_data = team_stats.get(away_team, {'AS': 10, 'AST': 4, 'Away_Form': 7})
        
        raw_hs = home_data.get('HS', 10)
        raw_hst = home_data.get('HST', 4)
        raw_h_form = home_data.get('Home_Form', 7) # ÚJ
        
        raw_as = away_data.get('AS', 10)
        raw_ast = away_data.get('AST', 4)
        raw_a_form = away_data.get('Away_Form', 7) # ÚJ
        
        # Fontos a sorrend: egyeznie kell a config.py FEATURES listájával!
        stats_array = [[raw_hs, raw_as, raw_hst, raw_ast, raw_h_form, raw_a_form]]
        scaled_stats = scaler.transform(stats_array)[0]
        
        # 0=HS, 1=AS, 2=HST, 3=AST, 4=Home_Form, 5=Away_Form
        features = [[h_id, a_id, scaled_stats[0], scaled_stats[1], scaled_stats[2], scaled_stats[3], scaled_stats[4], scaled_stats[5]]]
        
        active_model = models[selected_model_name]
        pred = active_model.predict(features)[0]
        
        res_map = {0: "Vendég (Away) győzelem 🔴", 1: "Döntetlen (Draw) ⚪", 2: "Hazai (Home) győzelem 🟢"}
        
        return jsonify({
            "status": "success",
            "prediction": res_map[int(pred)],
            "model_used": selected_model_name
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    app.run(port=8000, debug=True)