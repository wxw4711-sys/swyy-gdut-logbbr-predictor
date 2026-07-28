"""
logBBR 预测系统 - Gradio Web UI (Hugging Face Spaces 免费层)
复用 flask_server.py 的预测核心逻辑（模型 bundle / 特征工程 / 原子贡献溯源）。
"""
import traceback
import numpy as np
import pandas as pd
import gradio as gr

import flask_server as core  # 复用预测核心（不含 Flask 启动）

core.load_bundle()
bundle = core.bundle


def categorize(v):
    if v > 0.0:
        return "高 (骨>血)"
    if v > -0.1:
        return "中"
    return "低 (骨<血)"


def do_predict(smiles_list, animal_weights, genders, injection_dosages, true_logbbr=None):
    model = bundle["model"]
    scaler = bundle["scaler"]
    selected_indices = bundle["selected_indices"]
    full_feature_columns = bundle["full_feature_columns"]
    meta_defaults = bundle.get("meta_defaults", {})

    X_full = core.smiles_to_features(smiles_list, animal_weights, genders, injection_dosages, meta_defaults)
    X_full = X_full.reindex(columns=full_feature_columns, fill_value=0)
    X_sel = X_full.iloc[:, selected_indices].copy()
    X_scaled = scaler.transform(X_sel)
    pred = model.predict(X_scaled)

    importances = model.feature_importances_
    sel_cols = [full_feature_columns[i] for i in selected_indices]
    feat_name_map = bundle.get("feature_name_map", {})
    top_idx = np.argsort(importances)[::-1][:10]
    top_features = [
        {"name": feat_name_map.get(sel_cols[j], sel_cols[j]), "importance": round(float(importances[j]), 6)}
        for j in top_idx
    ]

    results = []
    for i, smi in enumerate(smiles_list):
        v = float(pred[i])
        kp = float(np.power(10.0, v))
        results.append({"index": i, "smiles": smi, "pred_logBBR": round(v, 4),
                        "pred_kP": round(kp, 4), "category": categorize(v)})
    results.sort(key=lambda x: x["pred_logBBR"], reverse=True)
    for nidx, r in enumerate(results):
        r["index"] = nidx

    metrics = {}
    if true_logbbr is not None and len(true_logbbr) == len(results) and len(true_logbbr) > 1:
        from scipy.stats import spearmanr, pearsonr
        y_true = np.array([float(x) for x in true_logbbr])
        y_pred = np.array([r["pred_logBBR"] for r in results])
        rho, _ = spearmanr(y_true, y_pred)
        pr, _ = pearsonr(y_true, y_pred)
        metrics = {"spearman": round(float(rho), 4), "pearson": round(float(pr), 4), "n": int(len(y_true))}

    return results, metrics, top_features


def run_single(smiles, weight, gender, dosage):
    if not smiles or not smiles.strip():
        return "请输入 SMILES"
    try:
        res, metrics, top = do_predict([smiles.strip()], [weight], [gender], [dosage])
        r = res[0]
        md = [
            "## 单分子预测结果", "",
            f"- **SMILES**: `{r['smiles']}`",
            f"- **预测 logBBR**: **{r['pred_logBBR']}**",
            f"- **骨-血分配系数 Kp = 10^logBBR**: **{r['pred_kP']}**",
            f"- **分类**: {r['category']}", "",
            "### Top 10 特征重要性",
        ]
        for f in top:
            md.append(f"- {f['name']}: {f['importance']}")
        return "\n".join(md)
    except Exception as e:
        return f"预测失败: {e}\n{traceback.format_exc()}"


def run_batch(text):
    if not text or not text.strip():
        return pd.DataFrame(), "请输入数据"
    rows = []
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 1 or not parts[0].strip():
            continue
        smi = parts[0].strip()
        weight = float(parts[1]) if len(parts) > 1 and parts[1].strip() else None
        gender = parts[2].strip() if len(parts) > 2 and parts[2].strip() else "male"
        dosage = float(parts[3]) if len(parts) > 3 and parts[3].strip() else None
        true = parts[4].strip() if len(parts) > 4 and parts[4].strip() else None
        rows.append((smi, weight, gender, dosage, true))
    if not rows:
        return pd.DataFrame(), "无有效数据"
    smiles_list = [r[0] for r in rows]
    weights = [r[1] for r in rows]
    genders = [r[2] for r in rows]
    dosages = [r[3] for r in rows]
    trues = [r[4] for r in rows]
    res, metrics, top = do_predict(smiles_list, weights, genders, dosages, trues)
    df = pd.DataFrame([{
        "SMILES": r["smiles"], "预测logBBR": r["pred_logBBR"],
        "Kp": r["pred_kP"], "分类": r["category"],
    } for r in res])
    md = "### 评估指标\n"
    if metrics:
        md += f"- Spearman ρ: {metrics['spearman']}\n- Pearson r: {metrics['pearson']}\n- 样本数: {metrics['n']}\n"
    else:
        md += "- 未提供实测 logBBR（最后一列），无法计算相关性指标。\n"
    return df, md


def run_structure(smiles):
    if not smiles or not smiles.strip():
        return "", pd.DataFrame(), "请输入 SMILES"
    try:
        smi = smiles.strip()
        model = bundle["model"]
        scaler = bundle["scaler"]
        selected_indices = bundle["selected_indices"]
        full_feature_columns = bundle["full_feature_columns"]
        meta_defaults = bundle.get("meta_defaults", {})
        feat_name_map = bundle.get("feature_name_map", {})

        X_full = core.smiles_to_features([smi], meta_defaults=meta_defaults)
        X_full = X_full.reindex(columns=full_feature_columns, fill_value=0)
        X_sel = X_full.iloc[:, selected_indices].copy()
        X_scaled = scaler.transform(X_sel)
        pred = model.predict(X_scaled)[0]
        raw_sel_row = X_sel.iloc[0].values

        atom_contrib, _active, mordred_details, atom_sources = core._compute_atom_contribs(
            smi, feat_name_map, selected_indices, full_feature_columns, model, raw_sel_row)
        mol_noh = core.Chem.RemoveHs(core.Chem.MolFromSmiles(smi))
        if mol_noh is None:
            return "", pd.DataFrame(), "无法解析 SMILES"
        highlight_atoms = list(atom_contrib.keys())
        atom_colors = {a: (1.0, max(0.0, 1.0 - 0.6 * c), max(0.0, 1.0 - 0.6 * c))
                       for a, c in atom_contrib.items()}
        svg = core._mol_to_svg(smi, highlight_atoms=highlight_atoms, atom_colors=atom_colors,
                               show_indices=True, size=(460, 460))
        atom_list = []
        for a, c in atom_contrib.items():
            atom_obj = mol_noh.GetAtomWithIdx(int(a))
            atom_list.append({
                "原子": f"{atom_obj.GetSymbol()}{int(a)}",
                "贡献": round(float(c), 4),
                "芳香性": bool(atom_obj.IsInRing()),
                "来源描述符": ", ".join(atom_sources.get(int(a), [])),
            })
        atom_df = pd.DataFrame(atom_list).sort_values("贡献", ascending=False)
        info = (f"预测 logBBR: **{round(float(pred), 4)}**  |  "
                f"Kp: **{round(float(np.power(10.0, pred)), 4)}**  |  "
                f"活跃原子数: {len(highlight_atoms)} / 重原子数: {int(mol_noh.GetNumHeavyAtoms())}")
        return (svg or ""), atom_df, info
    except Exception as e:
        return "", pd.DataFrame(), f"结构可视化失败: {e}\n{traceback.format_exc()}"


def model_info_md():
    rep = bundle.get("training_report", {})
    if not rep:
        return "模型信息不可用"
    return (
        "### 模型训练指标\n"
        f"- 测试集 R²: {rep.get('test_r2')}\n"
        f"- 测试集 RMSE: {rep.get('test_rmse')}\n"
        f"- 测试集 MAE: {rep.get('test_mae')}\n"
        f"- 交叉验证 R²: {rep.get('cv_r2_mean')} ± {rep.get('cv_r2_std')}\n"
        f"- 特征数: {rep.get('n_features')}\n"
        f"- 样本数: {rep.get('n_samples')}\n"
        f"- 模型类型: {bundle.get('model_type')}\n"
    )


with gr.Blocks(title="logBBR 骨血比预测系统 (LightGBM)") as demo:
    gr.Markdown("# logBBR 骨血比预测系统\n基于 LightGBM 的 QSAR 模型，输入 SMILES 与实验条件预测 logBBR（骨/血分配比）。")

    with gr.Tabs():
        with gr.Tab("单分子预测"):
            with gr.Row():
                with gr.Column():
                    s_smiles = gr.Textbox(label="SMILES", placeholder="输入分子 SMILES")
                    s_weight = gr.Number(label="动物体重 (g)", value=250)
                    s_gender = gr.Dropdown(["male", "female"], label="性别", value="male")
                    s_dosage = gr.Number(label="注射剂量", value=100)
                    s_btn = gr.Button("预测", variant="primary")
                with gr.Column():
                    s_out = gr.Markdown()
            s_btn.click(run_single, [s_smiles, s_weight, s_gender, s_dosage], s_out)

        with gr.Tab("批量预测"):
            b_in = gr.Textbox(label="批量输入（每行：SMILES\\t体重\\t性别\\t剂量\\t实测logBBR）",
                              lines=10, placeholder="SMILES\t250\tmale\t100\t0.5")
            b_btn = gr.Button("批量预测", variant="primary")
            b_df = gr.DataFrame()
            b_md = gr.Markdown()
            b_btn.click(run_batch, b_in, [b_df, b_md])

        with gr.Tab("结构可视化"):
            st_in = gr.Textbox(label="SMILES")
            st_btn = gr.Button("生成结构图", variant="primary")
            st_svg = gr.HTML()
            st_info = gr.Markdown()
            st_df = gr.DataFrame()
            st_btn.click(run_structure, st_in, [st_svg, st_df, st_info])

        with gr.Tab("模型信息"):
            gr.Markdown(model_info_md())

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
