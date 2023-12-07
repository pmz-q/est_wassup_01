import pandas as pd
from typing import Type


def custom_X_preprocess_cat(X_df: Type[pd.DataFrame]) -> Type[pd.DataFrame]:
  X_df = X_df.copy()
  
  # 전국 2019-2021 사고유형별 치사율 평균값
  # if '사고유형' in X_df.columns:
  #   X_df['사고유형'] = X_df['사고유형'].apply(lambda x:
  #     2.59 if x == '차대사람' else
  #     1.0 if x == '차대차' else
  #     14.57
  #   )
  
  # ['사고일시', '요일', '기상상태', '시군구', '도로형태', '노면상태', '사고유형']
  # 사고일시 yyyy-mm-dd hh datetime - 시간 컬럼 생성
  if '사고일시' in X_df.columns:
    X_df['사고일시'] = pd.to_datetime(X_df['사고일시'])
    X_df['시간'] = X_df['사고일시'].dt.hour
    X_df['월'] = X_df['사고일시'].dt.month
    X_df.drop(columns=['사고일시'], inplace=True)
    
  
  # 도로형태 e.g. 교차로 - 교차로안 -> 도로형태와 세부도로형태로 분리
    # 전국 도로형태별 2019 ~ 2021 치사율
  if '도로형태' in X_df.columns:
    X_df[['도로형태-1', '도로형태-2']] = X_df['도로형태'].str.split(' - ', expand=True)
    X_df.drop(columns=['도로형태'], inplace=True)
    # X_df['도로형태'] = X_df['도로형태'].str.split(' - ').apply(lambda x: 
    #     0.97 if x[1] == '교차로안' else
    #     0.94 if x[1] == '교차로부근' else
    #     3.68 if x[1] == '터널' else
    #     3.40 if x[1] == '교량' else
    #     1.88 if x[0] == '단일로' and x[1] == '기타' else
    #     1.99 if x[1] == '고가도로위' else
    #     3.10 if x[1] == '지하차도(도로)내' else
    #     2.00 if x[1] == '교차로횡단보도내' else
    #     1.18
    # )
    
  
  # 시군구 대전데이터 한정으로, 군, 동 정보만 남긴다.
  if '시군구' in X_df.columns:
    # 군, 구(동)
    X_df[['군', '구(동)']] = X_df['시군구'].str.split(' ', expand=True)[[1,2]]
    X_df.drop(columns=['시군구'], inplace=True)
    
    # df_test 의 unique values 와 df_train 의 unique values 가 다르므로 우선 제외시킴
    X_df.drop(columns=['구(동)'], inplace=True)
    # # 대구 군구별 치사율 2019~2021
    # X_df['시군구'] = X_df['시군구'].str.split(' ').apply(lambda x:
    #   1.14 if x[0] == '중구' else  
    #   1.01 if x[0] == '동구' else  
    #   0.74 if x[0] == '서구' else  
    #   0.55 if x[0] == '남구' else  
    #   0.83 if x[0] == '북구' else  
    #   0.43 if x[0] == '수성구' else  
    #   0.48 if x[0] == '달서구' else  
    #   2.10  
    # )
  
  if '노면상태' in X_df.columns:
    # 건조 젖음/습기 기타 서리/결빙 침수 적설
    X_df['노면상태'] = X_df['노면상태'].map(lambda x: 
      0 if x == '건조' else
      1 if x == '기타' else
      2 if x == '젖음/습기' else
      3 if x == '서리/결빙' else
      4
    )
  
  if '기상상태' in X_df.columns:
    # 맑음 비 흐림 기타 눈
    X_df['기상상태'] = X_df['기상상태'].map(lambda x: 
      0 if x == '맑음' else
      1 if x == '기타' else
      2 if x == '흐림' else
      3 if x == '비' else
      4
    )
  
  if '요일' in X_df.columns:
    # 월 화 수 목 금 토 일
    X_df['요일'] = X_df['요일'].map(lambda x: 
      0 if x == '월요일' else
      1 if x == '화요일' else
      2 if x == '수요일' else
      3 if x == '목요일' else
      4 if x == '금요일' else
      5 if x == '토요일' else
      6
    )
  
  return X_df

def merge_features_from_externals(X_df: Type[pd.DataFrame]) -> Type[pd.DataFrame]:
  X_df = X_df.copy()
  
  # main/data/origin/externals 에서 데이터 가져와서 추가 EDA 진행
  return X_df
