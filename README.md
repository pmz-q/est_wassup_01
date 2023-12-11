![middle](https://capsule-render.vercel.app/api?type=cylinder&color=0147FF&height=150&section=header&text=Wassup&fontColor=FFFFFF&fontSize=70&animation=fadeIn&fontAlignY=55)
<li align="center">
  
**WR(Water_Rocket)** <br>
김도연 : Team Leader 97 회로전공 ENTJ<br> 
김선들 : Git Branch management (96 컴퓨터전공 INTJ)<br>
김정현 : 01 소프트웨어전공 ISFJ<br>
박건수 : 96 정보통신공학전공 ISTP<br></li>
  
# prepocess_config .py  
  
DROP_X_COLS - train data에서 사용하지 않을 컬럼을 지정  
USE_X_COLS - train data에서 사용할 컬럼을 지정  
TARGET_COLS - train data의 target 컬럼을 지정  
  
+ config  
    + input_data  
        + train_csv : train data path  
        + test_csv : test data path  
  
    + output_data
        + train_feas_csv : 가공되어 저장 할 train data path  
        + test_feas_csv : 가공되어 저장 할 test data path  
        + train_target_csv : train data의 TARGET_COLS에서 지정한 컬럼만 추출하여 저장할 경로  
        + y_scaler_save : train_target에 사용한 scaler를 저장할 경로  
    
    
    + options  
        + index_col : train과 test에서 index로 사용할 컬럼 이름  
        + target_cols : TARGET_COLS  
        + ignore_drop_cols : True - 컬럼들을 drop 하지 않음, False - DROP_X_COLS의 컬럼들을 drop 함  
        + drop_cols : DROP_X_COLS  
        + use_cols : USE_X_COLS  
        + fill_num_strategy : null 값이 있는 부분을 채울 방법을 min, mean, max 중에서 선택  
        + x_scaler : SCALER['minmax']() put None for no X scaling process  
        + y_scaler : SCALER['minmax']() put None for no y scaling process  
  
------------  
  
  # config.py  
  
+ SCHEDULER
    + CAWarmRestarts: optim.lr_scheduler.CosineAnnealingWarmRestarts  
        + [참조링크](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)  
    + CALR: optim.lr_scheduler.CosineAnnealingLR  
        +  [참조링크](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html)  
    + lambdaLR: optim.lr_scheduler.LambdaLR  
        + [참조링크](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html)  
  
+ EXPERIMENT_NAME = output_files의 경로 중간의 삽입되는 이름  
  
+ config  
    + input_files  
        + X_csv : 가공한 train data path, preprocess의 output_data의 train_feas_csv 경로  
        + y_csv : 가공한 target data path, preprocess의 output_data의 train_target_csv 경로  
        + tst_csv : 가공한 test data path, preprocess의 output_data의 test_feas_csv 경로  
  
    + output_files  
        + output_train : 훈련 후 저장할 모델의 저장 할 경로  
        + output_eval : 훈련 후 저장할 평가지표 저장 할 경로  
        + 'output_pred : 훈련 후 test데이터로 예측한 결과 저장 할 경로  
  
    + model: ANN - 사용할 모델 class  
    + model_params  
        + input_dim: 'auto' - 입력한 데이터의 차원수를 자동으로 맞춰 줌  
        + hidden_dim : 훈련 시 hidden_dim의 차원과 레이어 수  
        + use_drop : drop out의 사용 유무 True - 사용, False - 사용 안함  
        + drop_ratio : drop out 사용 시 수치  
        + activation : 사용할 활성 함수 선택  
  
    + train_params  - 스케줄러 관한 설정  
        + use_scheduler: 스케줄러 사용 유/무 설정, True - 사용, False - 사용 안 함  
        + scheduler_cls: SCHEDULER['CALR']  
        + scheduler_params  
            + 'T_max': 200,  
            + 'eta_min': 0.0001  
  
        + loss : 사용할 손실 함수 선택  
        + optim : 사용할 옵티마이저 선택  
        + main_metric : 사용할 평가지표 선택, 밑의 metrics 중에서 선택, 다른걸 선택하고 싶다면 metrics에 추가 하여 사용  
        + metrics  
            'rmse': torchmetrics.MeanSquaredError(squared=False)  
            'mse': torchmetrics.MeanSquaredError()  
        + device : 연산할 device를 선택(cpu, cuda 등)  
        + epochs : train과 eval 실행 시 사용할 epoch 수  
        + data_loader_params  
            + batch_size : 배치사이즈 선택  
        + optim_params  
            + lr : learning rate 선택  
    + cv_params  
        + n_split : evaluation실행 시 사용할 fold 수 선택

---

<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white"></t><img src = "https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white">
