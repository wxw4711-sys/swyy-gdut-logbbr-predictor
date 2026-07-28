"""
logBBR 预测模型 - Flask 网页后端
支持模型训练与单分子/批量预测
"""
import os
import sys
import json
import base64
import io
import traceback
import warnings
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, send_from_directory

import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split, KFold, StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error, r2_score,
    mean_absolute_error, mean_absolute_percentage_error
)
from sklearn.feature_selection import (
    VarianceThreshold, SelectPercentile,
    RFE, mutual_info_regression
)
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from scipy.stats import spearmanr

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw, rdMolDescriptors
from rdkit import RDLogger
from mordred import Calculator, descriptors as mordred_descriptors

RDLogger.DisableLog('rdApp.*')

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ========== 配置 ==========
app = Flask(__name__, static_folder='static')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
RESULT_DIR = os.path.join(BASE_DIR, 'result')
CACHE_DIR = os.path.join(BASE_DIR, 'cache')

for d in [RESULT_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

BUNDLE_PATH = os.path.join(RESULT_DIR, 'logbbr_predictor_bundle.joblib')
DATA_PATH = os.path.join(BASE_DIR, '..', 'PTBD_v20240912.csv')

# 查找数据文件（部署到服务器时只需把 PTBD_v20240912.csv 放在项目根目录即可启用训练）
DATA_CANDIDATES = [
    DATA_PATH,
    os.path.join(BASE_DIR, "PTBD_v20240912.csv"),
    os.path.join(BASE_DIR, "..", "PTBD_v20240912.csv"),
]
DATA_FILE = None
for p in DATA_CANDIDATES:
    if os.path.exists(p):
        DATA_FILE = os.path.abspath(p)
        break

SEED = 42
N_FEATURES = 30
RFE_FEATURES = 30
N_OPTUNA_TRIAL = 50
CV_TIMES = 10
PRED_COL = "logbbr"
SMILE_COL = "SMILES"

# 全局变量
bundle = None
training_status = {"running": False, "progress": 0, "message": "", "error": None}

# ========== 日志配置 ==========
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("web_app")


# ========== 核心函数（从 notebook 移植）==========
def extract_isotope_strict(compound_index):
    if not isinstance(compound_index, str):
        return None
    if "18F" in compound_index.strip():
        return "18F"
    return None


def balance_18F_dataset_fixed(df, method="combined", seed=42):
    df = df.copy()
    df["has_18F"] = df["compound index"].apply(
        lambda x: extract_isotope_strict(x) == "18F" if pd.notna(x) else False
    )
    df["label_18F"] = df["has_18F"].astype(int)
    pos = df[df["label_18F"] == 1]
    neg = df[df["label_18F"] == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("无法平衡：某一类样本数为 0")
    if n_pos < n_neg:
        pos_extra = pos.sample(n=n_neg - n_pos, replace=True, random_state=seed)
        pos_bal = pd.concat([pos, pos_extra], ignore_index=True)
        neg_bal = neg
    elif n_neg < n_pos:
        neg_extra = neg.sample(n=n_pos - n_neg, replace=True, random_state=seed)
        neg_bal = pd.concat([neg, neg_extra], ignore_index=True)
        pos_bal = pos
    else:
        pos_bal, neg_bal = pos, neg
    balanced = pd.concat([pos_bal, neg_bal], ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return balanced


def calculate_morgan_fingerprints(smiles_list, radius=2, n_bits=1024):
    fingerprints = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fingerprints.append(np.zeros(n_bits))
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        fingerprints.append(arr)
    fp_df = pd.DataFrame(fingerprints)
    fp_df.columns = [f"Morgan_FP_{i}" for i in range(n_bits)]
    return fp_df


def calculate_Mordred_desc(smiles_list):
    calc = Calculator(mordred_descriptors, ignore_3D=True)
    all_descriptors = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            all_descriptors.append(np.zeros(len(calc.descriptors)))
        else:
            try:
                desc = calc(mol)
                all_descriptors.append(np.array(desc, dtype=float))
            except Exception:
                all_descriptors.append(np.zeros(len(calc.descriptors)))
    df = pd.DataFrame(all_descriptors)
    df.columns = [str(d) for d in calc.descriptors]
    df = df.fillna(0)
    return df


def build_full_features(df_balanced):
    y = df_balanced[PRED_COL].reset_index(drop=True)
    smiles = df_balanced[SMILE_COL].tolist()

    X_mordred = calculate_Mordred_desc(smiles)
    X_morgan = calculate_morgan_fingerprints(smiles)

    X_metadata = pd.DataFrame(index=df_balanced.index)
    X_metadata["is_rat"] = df_balanced["animal weight (g)"].apply(
        lambda x: 1 if pd.notna(x) and x > 100 else 0
    )
    X_metadata["is_male"] = df_balanced["gender"].apply(
        lambda x: 1 if isinstance(x, str) and x.lower() == "male" else 0
    )
    X_metadata["animal weight (g)"] = df_balanced["animal weight (g)"].fillna(
        df_balanced["animal weight (g)"].median()
    )
    X_metadata["injection_dosage"] = (
        df_balanced["injection dosage"]
        .fillna(df_balanced["injection dosage"].median())
        .clip(upper=500)
    )

    X = pd.concat([X_mordred, X_morgan, X_metadata.reset_index(drop=True)], axis=1)
    X.columns = X.columns.astype(str)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    return X, y


def adjusted_r2_score(r2, n, k):
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)


def smiles_to_features(smiles_list, animal_weights=None, genders=None,
                       injection_dosages=None, meta_defaults=None):
    """对输入的分子，按训练流程计算特征矩阵"""
    if meta_defaults is None:
        meta_defaults = {}
    smiles_list = list(smiles_list)
    n = len(smiles_list)
    aw = animal_weights if animal_weights is not None else [meta_defaults.get("animal_weight_median", 250)] * n
    gd = genders if genders is not None else [meta_defaults.get("default_gender", "male")] * n
    inj = injection_dosages if injection_dosages is not None else [meta_defaults.get("injection_dosage_median", 100)] * n

    X_mordred = calculate_Mordred_desc(smiles_list)
    X_morgan = calculate_morgan_fingerprints(smiles_list, radius=2, n_bits=1024)
    X_meta = pd.DataFrame(index=range(n))
    X_meta["is_rat"] = [1 if (pd.notna(w) and w > 100) else 0 for w in aw]
    X_meta["is_male"] = [1 if isinstance(g, str) and g.lower() == "male" else 0 for g in gd]
    X_meta["animal weight (g)"] = [(w if pd.notna(w) else meta_defaults.get("animal_weight_median", 250)) for w in aw]
    X_meta["injection_dosage"] = [min((inj_i if pd.notna(inj_i) else meta_defaults.get("injection_dosage_median", 100)), 500) for inj_i in inj]

    X = pd.concat([X_mordred, X_morgan, X_meta.reset_index(drop=True)], axis=1)
    X.columns = X.columns.astype(str)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    return X


# ========== 模型训练 ==========
def train_model():
    """完整训练流程"""
    global training_status, bundle
    training_status = {"running": True, "progress": 0, "message": "开始训练...", "error": None}

    try:
        if DATA_FILE is None:
            raise FileNotFoundError("未找到数据文件 PTBD_v20240912.csv")

        # 1. 读取数据
        training_status["progress"] = 5
        training_status["message"] = "读取数据集..."
        df = pd.read_csv(DATA_FILE, encoding="utf-8")
        df = df.dropna(subset=[PRED_COL])

        lower_pct = df[PRED_COL].quantile(0.05)
        upper_pct = df[PRED_COL].quantile(0.95)
        df = df[(df[PRED_COL] >= lower_pct) & (df[PRED_COL] <= upper_pct)]
        df = df[(df[PRED_COL] >= -0.5) & (df[PRED_COL] <= 1.5)]
        df = df.reset_index(drop=True)

        # 2. 18F 平衡
        training_status["progress"] = 10
        training_status["message"] = "18F 平衡处理..."
        df_balanced = balance_18F_dataset_fixed(df, method="combined", seed=SEED)

        # 3. 计算特征
        training_status["progress"] = 20
        training_status["message"] = "计算 Mordred 描述符 + Morgan 指纹 + Metadata 元数据（此步较慢，请耐心等待）..."
        X, y = build_full_features(df_balanced)

        # 建立特征名称映射
        readable_name_map = {}
        mordred_calc = Calculator(mordred_descriptors, ignore_3D=True)
        mordred_names = [str(d) for d in mordred_calc.descriptors]
        for i, col in enumerate(X.columns):
            col_str = str(col)
            if col_str.startswith("Morgan_FP_"):
                readable_name_map[col_str] = f"MorganFP_bit{col_str.replace('Morgan_FP_', '')}"
            elif col_str in ["is_rat", "is_male", "animal weight (g)", "injection_dosage"]:
                readable_name_map[col_str] = col_str
            elif i < len(mordred_names):
                readable_name_map[col_str] = f"Mordred_{mordred_names[i]}"
            else:
                readable_name_map[col_str] = col_str

        # 4. 特征筛选 VT -> TBE -> UFE -> RFE
        training_status["progress"] = 40
        training_status["message"] = "特征筛选 (VT → TBE → UFE → RFE)..."
        vt = VarianceThreshold(threshold=0.02)
        X_vt = vt.fit_transform(X)
        vt_mask = vt.get_support()

        max_tbe_features = min(2 * RFE_FEATURES, X_vt.shape[1])
        et = ExtraTreesRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
        et.fit(X_vt, y)
        tbe_indices = np.argsort(et.feature_importances_)[::-1][:max_tbe_features]
        tmp = np.where(vt_mask)[0]
        tbe_global_indices = tmp[tbe_indices]
        X_tbe = X_vt[:, tbe_indices]

        ufe = SelectPercentile(score_func=mutual_info_regression, percentile=80)
        X_ufe = ufe.fit_transform(X_tbe, y)
        ufe_local = np.where(ufe.get_support())[0]
        ufe_global_indices = tbe_global_indices[ufe_local]

        n_rfe = min(RFE_FEATURES, X_ufe.shape[1])
        rfe_estimator = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
        rfe = RFE(estimator=rfe_estimator, n_features_to_select=n_rfe, step=1)
        X_rfe = rfe.fit_transform(X_ufe, y)
        rfe_local = np.where(rfe.get_support())[0]
        selected_indices = sorted(list(set(ufe_global_indices[rfe_local])))

        if len(selected_indices) < N_FEATURES:
            et_all = ExtraTreesRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
            et_all.fit(X, y)
            remaining = [c for c in X.columns if c not in selected_indices]
            remaining_sorted = sorted(
                remaining,
                key=lambda c: et_all.feature_importances_[list(X.columns).index(c)],
                reverse=True
            )
            selected_indices.extend(remaining_sorted[:N_FEATURES - len(selected_indices)])

        selected_indices = sorted(list(set(selected_indices)))[:N_FEATURES]
        X = X.iloc[:, selected_indices].copy()

        # 5. 数据集划分
        training_status["progress"] = 50
        training_status["message"] = "数据集划分 (8:1:1)..."
        label_18F = df_balanced["label_18F"].reset_index(drop=True)
        y_bins = pd.cut(y, bins=4, labels=False, include_lowest=True)
        stratify_labels = y_bins.astype(str) + "_" + label_18F.astype(str)

        sss_test = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
        for train_val_idx, test_idx in sss_test.split(X, stratify_labels):
            train_val_idx = train_val_idx.tolist()
            test_idx = test_idx.tolist()

        relative_val_ratio = 0.1 / 0.9
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=relative_val_ratio, random_state=SEED)
        for train_idx, val_idx in sss_val.split(X.iloc[train_val_idx], stratify_labels.iloc[train_val_idx]):
            train_idx = [train_val_idx[i] for i in train_idx]
            val_idx = [train_val_idx[i] for i in val_idx]

        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_val = X.iloc[val_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_val = y.iloc[val_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # 6. Optuna 调优
        training_status["progress"] = 60
        training_status["message"] = f"Optuna 超参数调优 ({N_OPTUNA_TRIAL} trials)..."
        X_train_val = pd.concat([X_train, X_val], ignore_index=True)
        y_train_val = pd.concat([y_train, y_val], ignore_index=True)

        scaler_tv = MinMaxScaler()
        X_train_val_scaled = scaler_tv.fit_transform(X_train_val)
        X_train_val_df = pd.DataFrame(X_train_val_scaled, columns=X.columns)

        def objective(trial):
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_train_val_df, y_train_val, test_size=0.1, random_state=SEED
            )
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "random_state": SEED,
                "n_jobs": -1,
                "verbose": -1,
            }
            model = lgb.LGBMRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            y_pred = model.predict(X_te, num_iteration=model.best_iteration_)
            return r2_score(y_te, y_pred)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=N_OPTUNA_TRIAL)
        best_params = study.best_params

        # 7. 交叉验证
        training_status["progress"] = 75
        training_status["message"] = f"10 折交叉验证..."
        cv = KFold(n_splits=CV_TIMES, random_state=SEED, shuffle=True)
        cv_r2 = []
        for fold, (tr_idx, va_idx) in enumerate(cv.split(X_train_val_df, y_train_val)):
            m = lgb.LGBMRegressor(**best_params, random_state=SEED, n_jobs=-1, verbose=-1)
            m.fit(X_train_val_df.iloc[tr_idx], y_train_val.iloc[tr_idx],
                  eval_set=[(X_train_val_df.iloc[va_idx], y_train_val.iloc[va_idx])],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            pred = m.predict(X_train_val_df.iloc[va_idx], num_iteration=m.best_iteration_)
            cv_r2.append(r2_score(y_train_val.iloc[va_idx], pred))

        # 8. 最终模型训练
        training_status["progress"] = 85
        training_status["message"] = "训练最终模型..."
        final_model = lgb.LGBMRegressor(**best_params, random_state=SEED, n_jobs=-1, verbose=-1)
        final_model.fit(X_train_scaled, y_train,
                        eval_set=[(X_val_scaled, y_val)],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

        y_pred_test = final_model.predict(X_test_scaled, num_iteration=final_model.best_iteration_)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)

        # 9. 构建完整特征列
        _mordred_calc_full = Calculator(mordred_descriptors, ignore_3D=True)
        _full_feature_columns = (
            [str(d) for d in _mordred_calc_full.descriptors]
            + [f"Morgan_FP_{i}" for i in range(1024)]
            + ["is_rat", "is_male", "animal weight (g)", "injection_dosage"]
        )

        # 10. 保存 bundle
        training_status["progress"] = 95
        training_status["message"] = "保存模型..."
        bundle = {
            "model": final_model,
            "scaler": scaler,
            "selected_indices": [int(i) for i in selected_indices],
            "feature_name_map": readable_name_map,
            "full_feature_columns": _full_feature_columns,
            "morgan": {"radius": 2, "n_bits": 1024},
            "meta_defaults": {
                "animal_weight_median": float(df_balanced["animal weight (g)"].median()),
                "injection_dosage_median": float(df_balanced["injection dosage"].median()),
                "default_gender": "male",
            },
            "model_type": "LightGBM",
            "training_report": {
                "best_params": best_params,
                "test_rmse": float(test_rmse),
                "test_r2": float(test_r2),
                "test_mae": float(test_mae),
                "cv_r2_mean": float(np.mean(cv_r2)),
                "cv_r2_std": float(np.std(cv_r2)),
                "n_samples": int(len(df_balanced)),
                "n_features": int(len(selected_indices)),
            },
            "selected_features_readable": [
                readable_name_map.get(_full_feature_columns[i], _full_feature_columns[i])
                for i in selected_indices
            ],
        }
        joblib.dump(bundle, BUNDLE_PATH)

        training_status["progress"] = 100
        training_status["message"] = (
            f"训练完成！测试集 R² = {test_r2:.4f}, RMSE = {test_rmse:.4f}, "
            f"MAE = {test_mae:.4f}, CV R² = {np.mean(cv_r2):.4f}±{np.std(cv_r2):.4f}"
        )
        training_status["running"] = False
        log.info(training_status["message"])

    except Exception as e:
        training_status["running"] = False
        training_status["error"] = str(e)
        training_status["message"] = f"训练失败: {str(e)}"
        log.error(f"训练失败: {traceback.format_exc()}")


# ========== Flask 路由 ==========
@app.route('/')
def index():
    return send_from_directory(STATIC_DIR, 'index.html')


@app.route('/api/status', methods=['GET'])
def get_status():
    """获取模型状态"""
    model_exists = os.path.exists(BUNDLE_PATH)
    info = {"model_ready": model_exists, "training": training_status}
    if model_exists and bundle is None:
        load_bundle()
    if bundle:
        report = bundle.get("training_report", {})
        info["metrics"] = {
            "test_r2": report.get("test_r2"),
            "test_rmse": report.get("test_rmse"),
            "test_mae": report.get("test_mae"),
            "cv_r2_mean": report.get("cv_r2_mean"),
            "cv_r2_std": report.get("cv_r2_std"),
            "n_features": report.get("n_features"),
            "n_samples": report.get("n_samples"),
        }
        info["selected_features"] = bundle.get("selected_features_readable", [])
    return jsonify(info)


@app.route('/api/train', methods=['POST'])
def start_training():
    """启动模型训练"""
    global training_status
    if training_status["running"]:
        return jsonify({"success": False, "message": "训练已在进行中"})
    if DATA_FILE is None:
        return jsonify({"success": False, "message": "未找到数据文件，请在 DATA_CANDIDATES 中配置路径"})

    import threading
    t = threading.Thread(target=train_model, daemon=True)
    t.start()
    return jsonify({"success": True, "message": "训练已启动"})


@app.route('/api/train/progress', methods=['GET'])
def get_training_progress():
    """获取训练进度"""
    return jsonify(training_status)


def load_bundle():
    global bundle
    if os.path.exists(BUNDLE_PATH):
        bundle = joblib.load(BUNDLE_PATH)
        return True
    return False


@app.route('/api/predict', methods=['POST'])
def predict():
    """预测接口"""
    global bundle
    if bundle is None:
        if not load_bundle():
            return jsonify({"success": False, "message": "模型未训练，请先训练模型"})

    data = request.get_json()
    if not data or "smiles" not in data:
        return jsonify({"success": False, "message": "请提供 SMILES 列表"})

    smiles_list = data["smiles"]
    if isinstance(smiles_list, str):
        smiles_list = [smiles_list]

    animal_weights = data.get("animal_weights")
    genders = data.get("genders")
    injection_dosages = data.get("injection_dosages")
    # 可选：用户提供真实 logBBR，用于计算 Spearman ρ / Pearson 评估指标
    true_logbbr = data.get("true_logbbr")

    try:
        model = bundle["model"]
        scaler = bundle["scaler"]
        selected_indices = bundle["selected_indices"]
        full_feature_columns = bundle["full_feature_columns"]
        meta_defaults = bundle.get("meta_defaults", {})

        X_full = smiles_to_features(smiles_list, animal_weights, genders,
                                    injection_dosages, meta_defaults)
        X_full = X_full.reindex(columns=full_feature_columns, fill_value=0)
        X_sel = X_full.iloc[:, selected_indices].copy()
        X_scaled = scaler.transform(X_sel)
        pred = model.predict(X_scaled)

        # 分类
        def categorize(v):
            if v > 0.0:
                return "高 (骨>血)"
            if v > -0.1:
                return "中"
            return "低 (骨<血)"

        results = []
        for i, smi in enumerate(smiles_list):
            logbbr_i = float(pred[i])
            kp_i = float(np.power(10.0, logbbr_i))  # 骨-血分配系数 Kp = 10^logBBR
            results.append({
                "index": i,
                "smiles": smi,
                "pred_logBBR": round(logbbr_i, 4),
                "pred_kP": round(kp_i, 4),
                "category": categorize(logbbr_i),
            })

        # 特征重要性 Top 10
        importances = model.feature_importances_
        sel_cols = [full_feature_columns[i] for i in selected_indices]
        feat_name_map = bundle.get("feature_name_map", {})
        top_indices = np.argsort(importances)[::-1][:10]
        top_features = []
        for j in top_indices:
            top_features.append({
                "name": feat_name_map.get(sel_cols[j], sel_cols[j]),
                "importance": round(float(importances[j]), 6),
            })

        # 按预测 logBBR 从大到小排序
        results.sort(key=lambda x: x["pred_logBBR"], reverse=True)
        for new_idx, r in enumerate(results):
            r["index"] = new_idx

        # 若用户提供真实 logBBR，则计算 Spearman ρ 与 Pearson 相关指标
        metrics = {}
        if true_logbbr is not None:
            try:
                y_true = np.array([float(v) for v in true_logbbr], dtype=float)
                y_pred = np.array([r["pred_logBBR"] for r in results], dtype=float)
                if len(y_true) == len(y_pred) and len(y_true) > 1:
                    rho, _ = spearmanr(y_true, y_pred)
                    from scipy.stats import pearsonr
                    pr, _ = pearsonr(y_true, y_pred)
                    metrics = {
                        "spearman": round(float(rho), 4),
                        "pearson": round(float(pr), 4),
                        "n": int(len(y_true)),
                    }
            except Exception:
                metrics = {}

        return jsonify({
            "success": True,
            "results": results,
            "top_features": top_features,
            "metrics": metrics,
        })

    except Exception as e:
        return jsonify({"success": False, "message": f"预测失败: {str(e)}"})


@app.route('/api/feature_contrib', methods=['POST'])
def feature_contribution():
    """获取单个分子的特征贡献（用于可视化）"""
    global bundle
    if bundle is None:
        if not load_bundle():
            return jsonify({"success": False, "message": "模型未训练"})

    data = request.get_json()
    smiles = data.get("smiles", "")

    try:
        model = bundle["model"]
        scaler = bundle["scaler"]
        selected_indices = bundle["selected_indices"]
        full_feature_columns = bundle["full_feature_columns"]
        meta_defaults = bundle.get("meta_defaults", {})
        feat_name_map = bundle.get("feature_name_map", {})

        X_full = smiles_to_features([smiles], meta_defaults=meta_defaults)
        X_full = X_full.reindex(columns=full_feature_columns, fill_value=0)
        X_sel = X_full.iloc[:, selected_indices].copy()
        X_scaled = scaler.transform(X_sel)

        importances = model.feature_importances_
        sel_cols = [full_feature_columns[i] for i in selected_indices]
        scaled_vals = X_scaled[0]

        # 贡献度 ≈ importance × |scaled_value|
        contribs = importances * np.abs(scaled_vals)
        order = np.argsort(contribs)[::-1][:15]

        features = []
        for j in order:
            features.append({
                "name": feat_name_map.get(sel_cols[j], sel_cols[j]),
                "importance": round(float(importances[j]), 6),
                "value": round(float(scaled_vals[j]), 4),
                "contribution": round(float(contribs[j]), 6),
            })

        return jsonify({"success": True, "features": features})

    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


# ========== 分子结构与活性区域可视化辅助函数 ==========
def _mol_to_highlight_atoms(mol, active_bits, radius=2, n_bits=1024):
    """返回活跃 Morgan 比特对应的原子索引（用于结构高亮）。"""
    info = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, bitInfo=info)
    atoms = set()
    for b in active_bits:
        if b in info:
            for (aid, _r) in info[b]:
                atoms.add(aid)
    return list(atoms)


def _mol_to_svg(smiles, highlight_atoms=None, atom_colors=None, highlight_bonds=None,
                bond_colors=None, show_indices=True, size=(460, 460)):
    """
    将分子绘制为 SVG 字符串。
    - highlight_atoms/atom_colors: 高亮原子及其颜色
    - highlight_bonds/bond_colors: 高亮化学键（活性子结构骨架）
    - show_indices: 是否在每个原子旁标注原子编号（便于对应碳原子等）
    注意：编号使用去氢分子（重原子）的索引，与前端原子贡献列表一致。
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # 使用重原子分子（RemoveHs），使原子编号与贡献计算一致
    mol = Chem.RemoveHs(mol)
    try:
        drawer = Draw.rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
        opts = drawer.drawOptions()
        opts.clearBackground = False
        opts.addStereoAnnotation = True
        if show_indices:
            opts.addAtomIndices = True   # 显示原子编号
        opts.annotationFontScale = 0.75

        kwargs = {}
        if highlight_atoms:
            kwargs["highlightAtoms"] = list(highlight_atoms)
            if atom_colors:
                kwargs["highlightAtomColors"] = {a: atom_colors.get(a, (0.95, 0.4, 0.4))
                                                 for a in highlight_atoms}
        if highlight_bonds:
            kwargs["highlightBonds"] = list(highlight_bonds)
            if bond_colors:
                kwargs["highlightBondColors"] = bond_colors

        drawer.DrawMolecule(mol, **kwargs)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception:
        return None


def _bit_environment_atoms(mol, center_atom, radius):
    """获取以 center_atom 为中心、给定半径的子结构所覆盖的所有原子索引。"""
    if radius == 0:
        return {int(center_atom)}
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, int(center_atom))
    atoms = set()
    for bidx in env:
        bond = mol.GetBondWithIdx(bidx)
        atoms.add(bond.GetBeginAtomIdx())
        atoms.add(bond.GetEndAtomIdx())
    atoms.add(int(center_atom))
    return {int(a) for a in atoms}


def _mordred_atom_decompositions(mol_noh):
    """
    计算该分子每个重原子在关键 Mordred 描述符上的『原子级分解贡献』。
    这些分解值正是 Mordred 全局描述符（ALogP、AMR、TPSA 等）的原子组成来源，
    因此可作为『用 Mordred 描述符映射分子活性区域与结构』的锚点。

    返回:
      decomp : dict  轴名 -> list(每个重原子的分解贡献值，长度=重原子数)
    注：RDKit 的原子分解基于带氢分子，此处仅保留与 mol_noh 对齐的重原子（前 N 位）。
    """
    n_heavy = mol_noh.GetNumHeavyAtoms()
    mH = Chem.AddHs(mol_noh)
    decomp = {}
    try:
        crip = rdMolDescriptors._CalcCrippenContribs(mH)  # (ALogP, MR) 每原子
        decomp["亲脂性 (ALogP/SlogP)"] = [float(x[0]) for x in crip][:n_heavy]
        decomp["摩尔体积 (AMR)"] = [float(x[1]) for x in crip][:n_heavy]
    except Exception:
        pass
    try:
        tpsa = rdMolDescriptors._CalcTPSAContribs(mH)     # TPSA 每原子
        decomp["极性表面积 (TPSA)"] = [float(x) for x in tpsa][:n_heavy]
    except Exception:
        pass
    return decomp


# 每个 Mordred 描述符名前缀 -> 其原子级分解所归属的『理化属性轴』（对应 decomp 的键）
_AXIS_KEYWORDS = {
    "亲脂性 (ALogP/SlogP)": ["SlogP", "ALogP", "MLogP", "Crippen", "PEOE_VSA", "SMR", "BCUT",
                               "ATS", "ATSC", "GATS", "AATS", "VSA", "EState", "IC", "MIC",
                               "MAXs", "MAXss", "AMID", "MDEC", "MDEN", "SpAbs", "SpMax", "SpMean",
                               "MATS", "RDF", "Mor21", "Mor22", "Mor23", "Mor24", "Mor25", "Mor26",
                               "Mor27", "Mor28", "Mor29", "Mor30", "TOPO", "ETA", "P_VSA"],
    "摩尔体积 (AMR)": ["AMR", "CMR", "MolRef", "Refract", "SlogP_VSA", "SMR_VSA", "VSA_EState",
                        "BCUT", "ATS", "ATSC", "GATS", "AATS", "RDF", "Mor"],
    "极性表面积 (TPSA)": ["TPSA", "PSA", "Polar", "PEOE", "SlogP_VSA", "SMR_VSA", "VSA_EState",
                           "BCUT", "ATS", "ATSC", "GATS", "AATS", "RDF", "Mor", "EState"],
}


def _axis_for_mordred(desc_name):
    """
    返回该 Mordred 描述符所归属的理化属性轴（用于原子级溯源）。

    设计原则（修正版）：
    - 每个描述符只能归属到一个轴，按『最具体的物理含义』优先判定，避免互相矛盾。
    - 与 _AXIS_KEYWORDS 的归属保持一致（VSA 类、PEOE 属极性/体积，而非亲脂性）。
    - 自相关/拓扑描述符（AATS/ATSC/GATS/RDF/Mor/TOPO/ETA/MATS/Sp*/IC/MIC/MAXs/MINs/AMID/MDEC）
      本身不直接对应单一理化性质，默认归『亲脂性/疏水-电荷综合』轴，仅当名称显式含
      AMR/Refract/TPSA/PSA/Polar 等体积或极性关键词时才归到对应轴。
    """
    name = desc_name.lower()
    # 1) 最具体：摩尔体积相关（AMR/CMR/MolRef/Refract）
    if any(k.lower() in name for k in ["amr", "cmr", "molref", "refract"]):
        return "摩尔体积 (AMR)"
    # 2) 最具体：极性表面积相关（TPSA/PSA/Polar）
    if any(k.lower() in name for k in ["tpsa", "psa", "polar"]):
        return "极性表面积 (TPSA)"
    # 3) 体积型 VSA 描述符（SMR_VSA / SlogP_VSA 偏疏水体积，归摩尔体积轴）
    if "smr_vsa" in name or "slogp_vsa" in name:
        return "摩尔体积 (AMR)"
    # 4) 极性型 VSA / 电荷描述符（PEOE_VSA / VSA_EState / PEOE / EState 偏极性表面积轴）
    if "peoe_vsa" in name or "vsa_estate" in name or "peoe" in name or name.endswith("estate"):
        return "极性表面积 (TPSA)"
    # 5) BCUT：同时含体积与电荷信息，按主成分倾向（dv/pe/电荷）划到极性表面积轴
    if "bcut" in name:
        return "极性表面积 (TPSA)"
    # 6) 其余自相关/拓扑描述符（AATS/ATSC/GATS/RDF/Mor/TOPO/ETA/MATS/Sp*/IC/MIC/MAXs/MINs/AMID/MDEC 等）
    #    整体反映疏水-电荷综合性质，默认归『亲脂性 (ALogP/SlogP)』轴
    return "亲脂性 (ALogP/SlogP)"


def _compute_atom_contribs(smiles, feature_name_map, selected_indices, full_feature_columns,
                           model, X_scaled_row, morgan_cfg=None, top_n=10):
    """
    基于模型选中的 Mordred 描述符（其原子级分解贡献）按特征重要性加权，
    计算各重原子对预测活跃度的贡献，用于结构活性区域高亮与溯源。

    返回:
      atom_contrib : dict  atom_idx -> 归一化贡献强度 0~1
      active_bits  : list  占位（兼容旧签名，恒为空列表）
      mordred_details : list  每个主导 Mordred 描述符的原子级溯源明细
      atom_sources : dict  atom_idx -> 该原子来源的轴名列表
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}, [], [], {}
    mol_noh = Chem.RemoveHs(mol)
    n_atoms = mol_noh.GetNumAtoms()

    # 1. 每个重原子在关键 Mordred 描述符上的原子级分解贡献
    decomp = _mordred_atom_decompositions(mol_noh)

    # 2. 取得模型选中 Mordred 描述符的（标准化）特征重要性权重
    importances = model.feature_importances_
    sel_cols = [full_feature_columns[i] for i in selected_indices]

    # 描述符可读名 -> 加权重要性（重要性 x 标准化取值幅度）
    mordred_importance = {}
    for j, col in enumerate(sel_cols):
        fname = feature_name_map.get(col, col)
        if fname.startswith("Mordred_"):
            desc_name = fname[len("Mordred_"):]
            try:
                val = abs(float(X_scaled_row[j]))
            except Exception:
                val = 0.0
            mordred_importance[desc_name] = mordred_importance.get(desc_name, 0.0) + \
                float(importances[j]) * val

    # 3. 将每个 Mordred 描述符权重映射到其所属理化属性轴
    axis_weight = {}
    for desc_name, w in mordred_importance.items():
        axis = _axis_for_mordred(desc_name)
        if axis:
            axis_weight[axis] = axis_weight.get(axis, 0.0) + w

    # 4. 加权合成每个原子的贡献
    atom_raw = {}
    atom_sources = {}   # atom -> set of axis keys
    for dkey, per_atom in decomp.items():
        w = axis_weight.get(dkey, 0.0)
        # 若该轴未被模型选中（权重极低），给一个基础关注度，保证所有分子都能映射出活性区域
        eff_w = w if w > 1e-6 else 0.02
        for a, v in enumerate(per_atom):
            contrib = eff_w * abs(v)
            if contrib <= 0:
                continue
            atom_raw[a] = atom_raw.get(a, 0.0) + contrib
            atom_sources.setdefault(a, set()).add(dkey)

    if not atom_raw:
        return {}, [], [], {}

    # 5. 归一化原子贡献
    max_c = max(atom_raw.values())
    atom_contrib = {a: (v / max_c if max_c > 0 else 0) for a, v in atom_raw.items()}
    atom_sources = {int(a): sorted(list(bits)) for a, bits in atom_sources.items()}

    # 6. Mordred 描述符原子级溯源明细（列出每个理化轴贡献最大的原子）
    mordred_details = []
    # 若某理化轴无被模型选中的描述符，赋予『不同的』基础关注度，保证三类活性维度都能展示，
    # 且避免多个未命中轴因共用同一 base_w 而显示为完全相同的值。
    total_sel_w = sum(axis_weight.values())
    base_w = (total_sel_w / max(1, len(axis_weight))) * 0.1 if axis_weight else 0.02
    # 为三个轴分配互不相同的基础权重（亲脂性/摩尔体积/极性表面积）
    axis_base = {
        "亲脂性 (ALogP/SlogP)": base_w,
        "摩尔体积 (AMR)": base_w * 0.6,
        "极性表面积 (TPSA)": base_w * 0.35,
    }
    for dkey, per_atom in decomp.items():
        w = axis_weight.get(dkey, 0.0)
        if w <= 1e-6:
            w = axis_base.get(dkey, base_w)
        contrib_atoms = []
        for a, v in enumerate(per_atom):
            if abs(v) <= 0:
                continue
            c = w * abs(v)
            contrib_atoms.append((a, float(v), float(c)))
        contrib_atoms.sort(key=lambda x: x[2], reverse=True)
        mordred_details.append({
            "feature_name": dkey,
            "importance": round(float(w), 6),
            "is_selected_feature": w > 1e-6,
            "top_atoms": [
                {"atom": a, "value": round(val, 4), "contribution": round(c, 6)}
                for a, val, c in contrib_atoms[:8]
            ],
            "n_atoms": len(contrib_atoms),
        })
    mordred_details.sort(key=lambda x: x["importance"], reverse=True)

    return atom_contrib, [], mordred_details, atom_sources


@app.route('/api/structure', methods=['POST'])
def structure_visualization():
    """生成分子结构与活性区域可视化（高亮活跃子结构 + 原子贡献热力）。"""
    global bundle
    if bundle is None:
        if not load_bundle():
            return jsonify({"success": False, "message": "模型未训练"})

    data = request.get_json()
    smiles = (data.get("smiles") or "").strip()
    if not smiles:
        return jsonify({"success": False, "message": "请提供 SMILES"})

    try:
        model = bundle["model"]
        scaler = bundle["scaler"]
        selected_indices = bundle["selected_indices"]
        full_feature_columns = bundle["full_feature_columns"]
        meta_defaults = bundle.get("meta_defaults", {})
        feat_name_map = bundle.get("feature_name_map", {})

        # 计算特征与预测
        X_full = smiles_to_features([smiles], meta_defaults=meta_defaults)
        X_full = X_full.reindex(columns=full_feature_columns, fill_value=0)
        X_sel = X_full.iloc[:, selected_indices].copy()
        X_scaled = scaler.transform(X_sel)
        pred = model.predict(X_scaled)[0]

        importances = model.feature_importances_
        sel_cols = [full_feature_columns[i] for i in selected_indices]
        scaled_vals = X_scaled[0]
        contribs = importances * np.abs(scaled_vals)
        order = np.argsort(contribs)[::-1][:15]

        contrib_features = []
        for j in order:
            contrib_features.append({
                "name": feat_name_map.get(sel_cols[j], sel_cols[j]),
                "importance": round(float(importances[j]), 6),
                "value": round(float(scaled_vals[j]), 4),
                "contribution": round(float(contribs[j]), 6),
            })

        # 原子贡献 & Mordred 描述符原子级溯源（基于模型选中的 Mordred 描述符，按重要性加权）
        raw_sel_row = X_sel.iloc[0].values  # 原始（未缩放）选中特征
        atom_contrib, _active, mordred_details, atom_sources = _compute_atom_contribs(
            smiles, feat_name_map, selected_indices, full_feature_columns,
            model, raw_sel_row
        )

        mol_noh = Chem.RemoveHs(Chem.MolFromSmiles(smiles))
        highlight_atoms = list(atom_contrib.keys())

        # 原子颜色：红色系，贡献越高越红
        atom_colors = {}
        for a, c in atom_contrib.items():
            atom_colors[a] = (1.0, max(0.0, 1.0 - 0.6 * c), max(0.0, 1.0 - 0.6 * c))

        # 高亮活性子结构骨架中的化学键（两端原子都在高亮集合内）
        highlight_bonds = []
        bond_colors = {}
        hset = set(highlight_atoms)
        for bond in mol_noh.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a1 in hset and a2 in hset:
                bidx = bond.GetIdx()
                highlight_bonds.append(bidx)
                strength = (atom_contrib.get(a1, 0) + atom_contrib.get(a2, 0)) / 2.0
                bond_colors[bidx] = (1.0, max(0.0, 1.0 - 0.55 * strength),
                                     max(0.0, 1.0 - 0.55 * strength))

        svg = _mol_to_svg(smiles, highlight_atoms=highlight_atoms, atom_colors=atom_colors,
                          highlight_bonds=highlight_bonds, bond_colors=bond_colors,
                          show_indices=True, size=(460, 460))

        # 原子贡献列表（按贡献降序）：附带元素符号、成键氢数、来源 Mordred 描述符
        atom_list = []
        for a, c in atom_contrib.items():
            atom_obj = mol_noh.GetAtomWithIdx(int(a))
            atom_list.append({
                "atom": int(a),
                "symbol": atom_obj.GetSymbol(),
                "label": f"{atom_obj.GetSymbol()}{int(a)}",
                "n_hydrogens": int(atom_obj.GetTotalNumHs()),
                "in_ring": bool(atom_obj.IsInRing()),
                "is_aromatic": bool(atom_obj.GetIsAromatic()),
                "contrib": round(float(c), 4),
                "source_descs": atom_sources.get(int(a), []),
            })
        atom_list.sort(key=lambda x: x["contrib"], reverse=True)

        return jsonify({
            "success": True,
            "smiles": smiles,
            "pred_logBBR": round(float(pred), 4),
            "pred_kP": round(float(np.power(10.0, pred)), 4),
            "svg": svg,
            "atom_contrib": atom_list,
            "mordred_details": mordred_details,
            "n_active_atoms": len(highlight_atoms),
            "n_heavy_atoms": int(mol_noh.GetNumHeavyAtoms()),
            "contrib_features": contrib_features,
        })

    except Exception as e:
        return jsonify({"success": False, "message": f"结构可视化失败: {str(e)}"})


if __name__ == '__main__':
    # 启动时尝试加载已有模型
    load_bundle()
    if bundle:
        log.info("已加载已有模型")
    else:
        log.info("未找到已训练模型，请通过网页端点击训练按钮")

    # Hugging Face Spaces 通过环境变量 $PORT 指定对外端口（默认 7860）。
    # 本地运行时若未设置 PORT，则回退到 5000。
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
