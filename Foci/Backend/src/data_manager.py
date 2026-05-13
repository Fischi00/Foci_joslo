import pandas as pd
import glob
import os
from config import DATA_PATH

def load_all_football_data():
    all_files = glob.glob(os.path.join(DATA_PATH, "**/*.csv"), recursive=True)
    # HOZZÁADVA a 'Div' oszlop!
    desired_cols = ['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HS', 'AS', 'HST', 'AST']
    
    df_list = []
    for f in all_files:
        try:
            df = pd.read_csv(f, usecols=lambda c: c in desired_cols, encoding='latin1')
            df_list.append(df)
        except Exception as e:
            print(f"Hiba a fájl beolvasásakor ({f}): {e}")
            
    if not df_list:
        raise ValueError("Egyetlen fájlt sem sikerült beolvasni!")
        
    full_df = pd.concat(df_list, ignore_index=True)
    full_df['Date'] = pd.to_datetime(full_df['Date'], dayfirst=True, format='mixed', errors='coerce')
    
    return full_df.sort_values('Date')