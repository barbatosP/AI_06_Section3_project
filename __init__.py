from flask import Flask, jsonify
import sqlite3
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from category_encoders import OrdinalEncoder
import pandas as pd

app = Flask(__name__)


@app.route('/')
def index():
    #sqlite 데이터베이스 연결
    conn = sqlite3.connect('section3.db')
    cur = conn.cursor()

    #sqlite 데이터베이스에서 자료추출
    cur.execute("""SELECT *
                FROM rental_data_csv rdc 
                JOIN "temp" t ON rdc.대여일시 = t.date ;""")

    #dataframe 형식으로 자료 저장
    df_data = pd.DataFrame(cur.fetchall())
    df_data.drop([4,5], axis=1, inplace=True)
    df_data.columns = ["대여일시", "대여시간", "소재지", "대여건수", "온도"]

    target = "대여건수"
    features = df_data.columns.drop(target)

    #카테고리형 데이터를 인코딩
    enc=OrdinalEncoder()
    df_data = enc.fit_transform(df_data)

    #데이터를 train, test set 분리
    train, test = train_test_split(df_data, test_size=0.2)
    X_train = train[features]
    y_train = train[target]
    X_test = test[features]
    y_test = test[target]

    #머신러닝 모델 생성
    model = RandomForestRegressor(random_state=2,
                                    n_estimators=10,
                                    max_depth=16,
                                    max_features=0.5,
                                    min_samples_leaf=3)

    #모델 피팅
    model.fit(X_train, y_train)

    #target 예측 및 평가지표 확인
    y_pred_test = model.predict(X_test)

    #print("r2 score: ", r2_score(y_test, y_pred_test))
    #print("mse score: ", mean_squared_error(y_test, y_pred_test))
    return jsonify([r2_score(y_test, y_pred_test), mean_squared_error(y_test, y_pred_test)])
