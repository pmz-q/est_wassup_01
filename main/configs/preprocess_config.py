from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler

# Multi task Neural Network 을 위해선 하나 이상의 TARGET_COLS 를 적용해주세요.
# TARGET_COLS = ['사망자수', '중상자수', '경상자수', '부상자수']
# TARGET_DROP_COL = 'ECLO' # 없다면 None 값 주세요.
TARGET_COLS = ['ECLO']
TARGET_DROP_COL = None

# DROP_X_COLS 와 USE_X_COLS 는 둘 중 하나만 적용됩니다.
# ignore_drop_cols 에 따라서 결정됩니다.
DROP_X_COLS = []
USE_X_COLS = ['사고일시', '요일', '기상상태', '시군구', '도로형태', '노면상태', '사고유형']

# EMBEDDING_COLS 의 컬럼이 USE_X_COLS 에 들어있는지 반드시 확인해주세요.
# 만약, DROP_COLS 를 사용하는 경우, EMBEDDING COLS 가 DROP_COLS 에 포함되지 말아야 합니다.
# 만약, EMBEDDING_COLS 가 preprocess 과정에서 (custom_X_preprocess_cat 에서) 컬럼명이 변경된다면,
# (custom_X_preprocess_cat 에서) 변경되는 컬럼명으로 입력해주셔야 합니다.
# EMBEDDING_COLS = ['군']
EMBEDDING_COLS = []

# X 혹은 Y 의 scaler
# Custom scaler 의 경우, custom scaler class 를 생성하여 사용해주세요.
SCALER = {
  "standard": StandardScaler,
  "minmax": MinMaxScaler,
  "maxabs": MaxAbsScaler
}

config = {
  'input_data': {
    'train_csv': './data/origin/train.csv',
    'test_csv': './data/origin/test.csv'
  },
  'output_data': {
    'train_feas_csv': './data/features/train_X.csv',
    'test_feas_csv': './data/features/test_X.csv',
    'train_target_csv': './data/features/train_target.csv',
    'y_scaler_save': './data/features/y_scaler.save'
  },
  'options': {
    'embedding_cols': EMBEDDING_COLS, # [] for no embedding
    'index_col': 'ID',
    'target_cols': TARGET_COLS,
    'target_drop_col': TARGET_DROP_COL, # 없으면 None 값 주세요. target_cols 의 길이가 1 보다 클 때 적용됩니다.
    'ignore_drop_cols': True, # 'use_cols' is ignored when False, 'drop_cols' are ignored when True
    'drop_cols': DROP_X_COLS,
    'use_cols': USE_X_COLS,
    'fill_num_strategy': 'min', # choose one: ['min', 'mean', 'max']
    'x_scaler': SCALER['minmax'](), # put None for no X scaling process
    # 'y_scaler': None, # put None for no y scaling process
    'y_scaler': SCALER['minmax'](), # put None for no y scaling process
    # 'x_scaler': None
  }
}