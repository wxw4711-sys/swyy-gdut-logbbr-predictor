"""本地重建模型 bundle（部署到 PythonAnywhere 前必须执行一次）。

目的：用与服务器完全一致的 mordred 1.2.0 重新训练，生成
lightgbm.Booster + numpy 缩放参数的精简 bundle，使服务器预测接口
不再依赖 scikit-learn / scipy，从而塞进 512MB 免费磁盘。

用法（在 web_app 目录下，使用你本地装有 rdkit/mordred 1.2.0/sklearn/scipy/optuna 的环境）：
    python retrain_bundle.py
生成的 bundle 会写入 web_app/result/logbbr_predictor_bundle.joblib，
随后 git add/commit/push 即可。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import flask_server as fs  # noqa: E402


def main():
    print("=" * 60)
    print("开始训练（复用 flask_server.train_model）...")
    print("请确保本机 mordred 版本为 1.2.0：")
    import mordred
    print("  mordred version =", mordred.__version__)
    fs.train_model()
    print("=" * 60)
    print("完成。bundle 已保存至：", fs.BUNDLE_PATH)
    if not os.path.exists(fs.BUNDLE_PATH):
        print("警告：未找到 bundle 文件，请检查训练是否成功。")
        sys.exit(1)


if __name__ == "__main__":
    main()
