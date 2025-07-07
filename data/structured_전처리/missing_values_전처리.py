from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings
import sklearn
from packaging import version
from sklearn.metrics import mutual_info_score
warnings.filterwarnings("ignore")
# 1. 데이터 로드
df = pd.read_csv('250530_output.csv')



# 2. 결측치가 있는 컬럼만 선택
missing_cols = df.columns[df.isnull().any()].tolist()

# 3. 버전 호환 OneHotEncoder
if version.parse(sklearn.__version__) >= version.parse("1.2"):
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
else:
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

# 4. 컬럼별 결측치 채우기
for i, target_col in enumerate(tqdm(missing_cols, desc="🔧 전체 진행률")):
    print(f"\n[{i+1}/{len(missing_cols)}] ➡️ {target_col} 처리 중...")

    sub_tasks = ["데이터 분리", "전처리 파이프라인 구성", "모델 생성", "모델 학습", "예측 및 채우기"]
    for step in tqdm(sub_tasks, desc=f"⚙️ {target_col} 처리 단계", leave=False):
        try:
            if step == "데이터 분리":
                # 1) 결측치 없는 연속형 변수만 추출하여 상관관계 계산
                numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
                corr_candidates = [col for col in numeric_cols if col != target_col and df[col].isnull().sum() == 0]
                
                if len(corr_candidates) == 0:
                    print(f"⚠️ {target_col} → 상관관계 계산 가능한 연속형 변수 없음")
                    break

                # 상관계수 절댓값 기준으로 top 20 추출
                corrs = df[corr_candidates].corrwith(df[target_col]).abs().sort_values(ascending=False)
                top20_cols = corrs.head(20).index.tolist()

                feature_cols = top20_cols.copy()

                if len(feature_cols) == 0:
                    print(f"⚠️ {target_col} → usable feature 없음 (건너뜀)")
                    break

                # 2) 데이터 타입별로 나누기
                cat_cols = [col for col in feature_cols if df[col].dtype == 'object' or df[col].dtype.name == 'category']
                num_cols = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']]

                train_df = df[df[target_col].notna()]
                X_train = train_df[feature_cols]
                y_train = train_df[target_col]
                X_pred = df[df[target_col].isna()][feature_cols]
                pred_idx = df[df[target_col].isna()].index

            elif step == "전처리 파이프라인 구성":
                preprocessor = ColumnTransformer([
                    ('cat', encoder, cat_cols),
                    ('num', 'passthrough', num_cols)
                ])

            elif step == "모델 생성":
                if df[target_col].dtype in ['float64', 'int64']:
                    model = make_pipeline(
                        preprocessor,
                        LGBMRegressor(
                            random_state=42,
                            device="gpu",
                            n_estimators=100,
                            verbose=-1
                        )
                    )
                else:
                    model = make_pipeline(
                        preprocessor,
                        LGBMClassifier(
                            random_state=42,
                            device="gpu",
                            n_estimators=100,
                            verbose=-1
                        )
                    )

            elif step == "모델 학습":
                model.fit(X_train, y_train)

            elif step == "예측 및 채우기":
                y_pred = model.predict(X_pred)

                # 연속형 변수는 반올림
                if df[target_col].dtype in ['float64', 'int64']:
                    y_pred = np.round(y_pred)

                df.loc[pred_idx, target_col] = y_pred
                print(f"✅ {target_col} 결측치 {len(pred_idx)}개 채움")

        except Exception as e:
            print(f"❌ {target_col} - 단계 '{step}' 중 오류 발생: {e}")
            break
df.to_csv("final_df_250530.csv", index=False)
print("\n💾 저장 완료: final_df_filled.csv")



import pandas as pd
import numpy as np
from dython.nominal import theils_u
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# 데이터 로드
df = pd.read_csv("final_df_250530.csv")

# 제거할 열 리스트
erase = [
    'jobwave', 'version', 'hhid26_x', 'hmem26_x', 'hwaveent',
    'p260104', 'p260105', 'p260106', 'p263401'
]

# 제거 수행 (존재할 때만)
df = df.drop(columns=[col for col in erase if col in df.columns])

# 수치형 변수 리스트
numeric_features = [
    'h260150', 'h260301', 'h260302', 'h260303', 'h260304', 'h260741', 'h260742', 'h260743', 'h260744',
    'h260761', 'h260762', 'h260763', 'h260764', 'h261302', 'h261314', 'h261315', 'h261408', 'h261409',
    'h261410', 'h261411', 'h261412', 'h261413', 'h261414', 'h262102', 'h262134', 'h262153', 'h262153a',
    'h262158', 'h262202', 'h262210', 'h262301', 'h262311', 'h262312', 'h262313', 'h262314', 'h262315',
    'h262316', 'h262317', 'h262318', 'h262319', 'h262320', 'h262321', 'h262323', 'h262324', 'h262325',
    'h262326', 'h262327', 'h262328', 'h262329', 'h262332', 'h262330', 'h262331', 'h262402', 'h262412',
    'h262416', 'h262425', 'h262562', 'h262566', 'h262602', 'h262603', 'h262653', 'j205', 'j206',
    'j730', 'j316', 'j318', 'j322', 'j325', 'j609', 'j610', 'p260107', 'p260114', 'p260115', 'p260301',
    'p260302', 'p260303', 'p260402', 'p260405', 'p260407', 'p261006', 'p261007', 'p261031', 'p261032',
    'p261642', 'p261643', 'p261662', 'p261692', 'p261693', 'p261672', 'p261702', 'p261703', 'p262143',
    'p262152', 'p262153', 'p262154', 'p262160', 'p264160', 'p264161', 'p264164', 'p264172', 'p264414',
    'p265155', 'p265156', 'p265157', 'p265908', 'p265909', 'p265910', 'p265911', 'pa265908', 'pa265909',
    'pa265910', 'pa265911', 'pa269501', 'pa269502', 'pa269507', 'pa269508', 'w26p_l', 'w26p_c',
    'sw26p_l', 'sw26p_c', 'nw26p_l', 'nw26p_c', 'version_y'
]

def normalized_mi(x, y):
    mi = mutual_info_score(x, y)
    h_x = mutual_info_score(x, x)
    h_y = mutual_info_score(y, y)
    return mi / max(h_x, h_y) if max(h_x, h_y) != 0 else 0.0

# 📊 상관관계 계산 함수
def separate_and_compute_correlations(df: pd.DataFrame, numeric_features: list):
    numeric_cols = [col for col in numeric_features if col in df.columns]
    categorical_cols = [col for col in df.columns if col not in numeric_cols]

    # Pearson correlation
    print("📈 Pearson correlation 계산 중...")
    numeric_corr = pd.DataFrame(index=numeric_cols, columns=numeric_cols)
    for col1 in tqdm(numeric_cols, desc="Pearson Rows"):
        for col2 in numeric_cols:
            try:
                corr = df[[col1, col2]].corr(method='pearson').iloc[0, 1]
                numeric_corr.loc[col1, col2] = round(corr, 4)
            except:
                numeric_corr.loc[col1, col2] = None
    numeric_corr = numeric_corr.astype(float)

    # Mutual Information (normalized) for categorical vars
    print("🔍 Mutual Information 계산 중...")
    categorical_corr = pd.DataFrame(index=categorical_cols, columns=categorical_cols)
    for col1 in tqdm(categorical_cols, desc="Mutual Info Rows"):
        for col2 in categorical_cols:
            if col1 == col2:
                categorical_corr.loc[col1, col2] = 1.0
            else:
                try:
                    score = normalized_mi(df[col1], df[col2])
                    categorical_corr.loc[col1, col2] = round(score, 4)
                except:
                    categorical_corr.loc[col1, col2] = None
    categorical_corr = categorical_corr.astype(float)

    return numeric_cols, categorical_cols, numeric_corr, categorical_corr

# 실행
num_cols, cat_cols, num_corr, cat_corr = separate_and_compute_correlations(df, numeric_features)

# 출력
print("\n✅ 수치형 변수 간 Pearson 상관관계:")
print(num_corr)

print("\n✅ 범주형 변수 간 Mutual Information (normalized):")
print(cat_corr)

# 저장
num_corr.to_csv("pearson_correlation_matrix.csv")
cat_corr.to_csv("mutual_info_correlation_matrix.csv")
print("\n💾 저장 완료:")
print(" - pearson_correlation_matrix.csv")
print(" - mutual_info_correlation_matrix.csv")
