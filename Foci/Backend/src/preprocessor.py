import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os
from config import MODEL_PATH

class FootballPreprocessor:
    def __init__(self):
        self.team_le = LabelEncoder()
        self.target_le = LabelEncoder()
        self.scaler = StandardScaler()

    def fit_encoders(self, df):
        df = df.sort_values('Date')
        all_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).astype(str).unique()
        self.team_le.fit(all_teams)
        self.target_le.fit(['H', 'D', 'A'])
        
        teams_by_league = {}
        league_map = {
            'E0': '🇬🇧 Premier League (Első osztály)',
            'E1': '🇬🇧 Championship (Másodosztály)',
            'E2': '🇬🇧 League One (Harmadosztály)',
            'E3': '🇬🇧 League Two (Negyedosztály)',
            'EC': '🇬🇧 National League / Conference (Ötödosztály)',
            'SC0': '🇬🇧 Premiership (Első osztály)',
            'SC1': '🇬🇧 Championship (Másodosztály)',
            'SC2': '🇬🇧 League One (Harmadosztály)',
            'SC3': '🇬🇧 League Two (Negyedosztály)',
            'D1': '🇩🇪 Bundesliga 1 (Első osztály)',
            'D2': '🇩🇪 Bundesliga 2 (Másodosztály)',
            'I1': '🇮🇹 Serie A (Első osztály)',
            'I2': '🇮🇹 Serie B (Másodosztály)',
            'SP1': '🇪🇸 La Liga / Primera División (Első osztály)',
            'SP2': '🇪🇸 Segunda División (Másodosztály)',
            'F1': '🇫🇷 Ligue 1 (Első osztály)',
            'F2': '🇫🇷 Ligue 2 (Másodosztály)',
            'N1': '🇳🇱 Eredivisie (Első osztály)',
            'B1': '🇧🇪 Jupiler Pro League (Első osztály)',
            'P1': '🇵🇹 Primeira Liga (Első osztály)',
            'T1': '🇹🇷 Süper Lig (Első osztály)',
            'G1': '🇬🇷 Szuperliga (Első osztály)'
        }
        if 'Div' in df.columns:
            for div in df['Div'].dropna().unique():
                league_name = league_map.get(str(div).strip(), f"Egyéb Liga ({div})")
                teams_in_div = sorted(df[df['Div'] == div]['HomeTeam'].dropna().unique().tolist())
                if teams_in_div:
                    teams_by_league[league_name] = teams_in_div
        else:
            teams_by_league["Összes csapat"] = sorted(all_teams.tolist())

        # --- ÚJ RÉSZ: FORMA (Utolsó 5 meccs) SZÁMÍTÁSA ---
        df['Home_Points'] = df['FTR'].map({'H': 3, 'D': 1, 'A': 0})
        df['Away_Points'] = df['FTR'].map({'A': 3, 'D': 1, 'H': 0})
        
        team_stats = {}
        for team in all_teams:
            home_games = df[df['HomeTeam'] == team]
            away_games = df[df['AwayTeam'] == team]
            
            team_stats[team] = {
                'HS': home_games['HS'].mean() if not home_games.empty else 10,
                'HST': home_games['HST'].mean() if not home_games.empty else 4,
                'AS': away_games['AS'].mean() if not away_games.empty else 10,
                'AST': away_games['AST'].mean() if not away_games.empty else 4,
                # Kiszámoljuk a csapat JELENLEGI formáját a React számára
                'Home_Form': home_games['Home_Points'].tail(5).sum() if not home_games.empty else 7,
                'Away_Form': away_games['Away_Points'].tail(5).sum() if not away_games.empty else 7
            }
        
        with open(os.path.join(MODEL_PATH, 'team_encoder.pkl'), 'wb') as f:
            pickle.dump(self.team_le, f)
        with open(os.path.join(MODEL_PATH, 'team_stats.pkl'), 'wb') as f:
            pickle.dump(team_stats, f)
        with open(os.path.join(MODEL_PATH, 'teams_by_league.pkl'), 'wb') as f:
            pickle.dump(teams_by_league, f)
        
        # --- ÚJ RÉSZ: Legutóbbi egymás elleni meccsek kimentése validációhoz (JAVÍTOTT, GOLYÓÁLLÓ VERZIÓ) ---
        h2h_matches = {}
        hg_col = 'FTHG' if 'FTHG' in df.columns else None
        ag_col = 'FTAG' if 'FTAG' in df.columns else None
        
        for (h_team, a_team), group in df.groupby(['HomeTeam', 'AwayTeam']):
            last_match = group.iloc[-1] # A legutolsó meccsük
            
            # Csak akkor alakítjuk int-é, ha a gól oszlop létezik ÉS nem NaN (hiányzó) az értéke
            if hg_col and ag_col and not pd.isna(last_match[hg_col]) and not pd.isna(last_match[ag_col]):
                score_str = f"{int(last_match[hg_col])} - {int(last_match[ag_col])}"
            else:
                score_str = "? - ?"
            
            h2h_matches[f"{h_team} vs {a_team}"] = {
                'FTR': str(last_match['FTR']).strip().upper() if not pd.isna(last_match['FTR']) else "D",
                'Score': score_str,
                'Date': str(last_match['Date']) if 'Date' in group.columns else "Ismeretlen dátum"
            }
            
        with open(os.path.join(MODEL_PATH, 'h2h_matches.pkl'), 'wb') as f:
            pickle.dump(h2h_matches, f)


    def transform_data(self, df, is_training=True):
        df = df.dropna(subset=['HomeTeam', 'AwayTeam', 'FTR']).copy()
        df['FTR'] = df['FTR'].astype(str).str.strip().str.upper()
        df = df[df['FTR'].isin(['H', 'D', 'A'])].copy()

        df = df.sort_values('Date')
        
        # Múltbeli meccsek formájának számítása a tanításhoz (Gördülő összeadás)
        df['Home_Points'] = df['FTR'].map({'H': 3, 'D': 1, 'A': 0})
        df['Away_Points'] = df['FTR'].map({'A': 3, 'D': 1, 'H': 0})
        df['Home_Form'] = df.groupby('HomeTeam')['Home_Points'].transform(lambda x: x.rolling(5, min_periods=1).sum())
        df['Away_Form'] = df.groupby('AwayTeam')['Away_Points'].transform(lambda x: x.rolling(5, min_periods=1).sum())

        df['Home_Id'] = self.team_le.transform(df['HomeTeam'].astype(str))
        df['Away_Id'] = self.team_le.transform(df['AwayTeam'].astype(str))
        df['Target'] = self.target_le.transform(df['FTR'])
        
        df = df.fillna(0)
        
        # A forma bekerült a normalizálóba!
        stats_cols = ['HS', 'AS', 'HST', 'AST', 'Home_Form', 'Away_Form']
        if is_training:
            df[stats_cols] = self.scaler.fit_transform(df[stats_cols])
            with open(os.path.join(MODEL_PATH, 'scaler.pkl'), 'wb') as f:
                pickle.dump(self.scaler, f)
        else:
            df[stats_cols] = self.scaler.transform(df[stats_cols])
            
        return df