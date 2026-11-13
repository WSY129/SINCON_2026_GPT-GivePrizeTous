from pathlib import Path

import numpy as np
import pandas as pd

# Kaggle Notebook이면 /kaggle/input 아래에서, 로컬이면 data/ 폴더에서 읽습니다.
kaggle_files = list(Path('/kaggle/input').glob('**/train.csv')) if Path('/kaggle/input').exists() else []
DATA_DIR = kaggle_files[0].parent if kaggle_files else Path('data')

train = pd.read_csv(DATA_DIR / 'train.csv')
test = pd.read_csv(DATA_DIR / 'test.csv')
print(train.shape, test.shape)


def recent_profile(history, start, end):
    """start~end 기간의 역·요일·시간대별 log(1+승차) 평균"""
    window = history[(history['date'] >= start) & (history['date'] <= end)].copy()
    window['weekday'] = pd.to_datetime(window['date']).dt.weekday  # 0=월 ... 6=일
    window['log_boardings'] = np.log1p(window['boardings'])
    return window.groupby(['station_id', 'weekday', 'hour'])['log_boardings'].mean()


def predict(profile, rows):
    """각 행의 (역, 요일, 시간)으로 평균을 찾아 승차 인원 단위로 되돌립니다."""
    keys = pd.MultiIndex.from_arrays([rows['station_id'], pd.to_datetime(rows['date']).dt.weekday, rows['hour']])
    return np.expm1(profile.reindex(keys).to_numpy())


def rmsle(pred, actual):
    return np.sqrt(np.mean((np.log1p(pred) - np.log1p(actual)) ** 2))


# 1) 자체 검증: 2026-01-05 ~ 02-28로 만든 평균으로 3~6월을 예측해 실제값과 비교합니다.
valid = train[(train['date'] >= '2026-03-01') & (train['date'] <= '2026-06-30')]
valid_pred = predict(recent_profile(train, '2026-01-05', '2026-02-28'), valid)
print('검증 RMSLE:', round(rmsle(valid_pred, valid['boardings']), 5))

# 2) 제출 파일: 6월 30일까지의 최근 8주로 7~10월 전체를 예측합니다.
test['boardings'] = predict(recent_profile(train, '2026-05-04', '2026-06-30'), test)

submission = test[['example_id', 'boardings']]
assert len(submission) == 157194 and submission['boardings'].notna().all()
submission.to_csv('submission.csv', index=False)
print(submission.head())
