import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'main')
MODEL_PATH = os.path.join(BASE_DIR, 'models')

FEATURES = ['Home_Id', 'Away_Id', 'HS', 'AS', 'HST', 'AST', 'Home_Form', 'Away_Form']
TARGET = 'FTR' # Full Time Result (H, D, A)