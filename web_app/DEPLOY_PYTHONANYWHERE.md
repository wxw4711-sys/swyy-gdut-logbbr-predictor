# 部署到 PythonAnywhere 免费版（WSGI / Flask）

本应用是**服务端 Python 程序**，依赖 `rdkit` + `mordred` + `lightgbm` 做分子描述符计算与预测。
PythonAnywhere 免费版**无需信用卡**，但有以下限制：

- 免费账户域名固定为 `<你的用户名>.pythonanywhere.com`（用户名即账户名，注册时选定）。
- 仅 512MB 磁盘、有限 CPU 与每日带宽额度；**长时间无访问后会休眠**，首次访问需冷启动数秒~数十秒。
- Web 应用只能访问白名单网络（PyPI 可装包），**运行时不需要联网**（模型已打包在 `result/` 内）。

> 推荐注册用户名含 `swyygdutlogbbr`，得到 `swyygdutlogbbr.pythonanywhere.com`。

---

## 步骤 0（本机，部署前必须）：用 mordred 1.2.0 重建模型 bundle

> 服务器上跑的是 **mordred 1.2.0**，而旧 bundle 是用旧版 `mordred-ojmb` 训练的。
> 两者算出的描述符列不一致会导致预测漂移，且旧 bundle 保存的是 `sklearn` 模型，
> 反序列化需要 `scikit-learn`/`scipy`（免费版装不下）。
>
> 因此**部署前必须在本机用 `mordred==1.2.0` 重新训练一次**，生成
> `lightgbm.Booster + numpy 缩放参数` 的精简 bundle（预测接口不再依赖 sklearn/scipy）。

在你本地装有 `rdkit` / `scikit-learn` / `scipy` / `optuna` 的 Python 环境里操作：

```bash
# 1) 临时切到与服务器一致的 mordred 版本（如本机是 mordred-ojmb 需先卸）
pip uninstall -y mordred-ojmb
pip install "mordred==1.2.0" "networkx==2.8.8"

# 2) 进入 web_app 目录，运行重训脚本
cd web_app
python retrain_bundle.py
# 脚本会打印 "mordred version = 1.2.0" 并训练，最终把 bundle 写入
# web_app/result/logbbr_predictor_bundle.joblib

# 3) 提交并推送到 GitHub
git add result/logbbr_predictor_bundle.joblib
git commit -m "retrain bundle with mordred 1.2.0 (sklearn-free Booster)"
git push

# 4) （可选）把本机 mordred 还原回你原本的版本，不影响部署
```

> 重训脚本 `web_app/retrain_bundle.py` 复用 `flask_server.train_model()`，
> 只是把保存格式从 `sklearn.LGBMRegressor + MinMaxScaler` 改为
> `lightgbm.Booster + numpy 缩放参数`，训练逻辑完全一致。

---

## 步骤 1：注册
1. 打开 https://www.pythonanywhere.com/ ，点 **Create a free account**。
2. 用户名建议填 `swyygdutlogbbr`（决定域名），邮箱、密码填好提交。
3. 登录后进入 Dashboard。

## 步骤 2：把代码放进 PythonAnywhere（二选一）

### 方式 A（推荐，最简单）：从 GitHub 克隆
1. Dashboard 顶部点 **Consoles** → **Bash**（启动一个命令行）。
2. 执行：
   ```bash
   git clone https://github.com/wxw4711-sys/swyy-gdut-logbbr-predictor.git
   ```
   这会在你的家目录生成 `swyy-gdut-logbbr-predictor/` 文件夹（含**步骤 0 重新生成的**模型 bundle）。

### 方式 B：上传 ZIP
1. 在本机把 `web_app/` 文件夹压缩成 `web_app.zip`。
2. Dashboard 顶部点 **Files**，进入 `/home/你的用户名/`，点 **Upload a file** 上传 zip，
   然后到 **Consoles → Bash** 解压：
   ```bash
   unzip web_app.zip -d swyy-gdut-logbbr-predictor
   ```

## 步骤 3：安装依赖（免费版关键：只装 numpy + pandas，绝不装 scipy/sklearn）
> **重要更正**：PythonAnywhere 免费版系统 Python 3.8 **并不自带** numpy/scipy/pandas/sklearn（实测缺失）。
> 但 `scipy`/`scikit-learn` 各自约 200MB，**装了必超 512MB**。
> 本仓库的预测接口已改为**只依赖 `numpy + pandas`**（不再依赖 scipy/sklearn），
> 所以只需把 `rdkit/mordred/lightgbm` 等重包（已装进 `~/.local`）和 `numpy/pandas` 凑齐即可。
>
> 你之前已经把 `rdkit-pypi`/`mordred`/`lightgbm`/`joblib`/`six`/`networkx`/`flask` 装进了 `~/.local`（约 303MB），
> **不要 `rm -rf ~/.local`**，只需在其上补装 `numpy` + `pandas`。

在 **Consoles → Bash** 中：
```bash
cd ~/swyy-gdut-logbbr-predictor
git pull                       # 拉取步骤 0 的新 bundle 与新 flask_server.py

# 只装预测必需的 numpy + pandas（pip 会自动带上 python-dateutil/pytz 等小依赖，不会拉 scipy）
pip3.8 install --user --no-cache-dir numpy pandas
```

安装完成后**务必验证**（免费版最容易卡在这里）：
```bash
python3.8 -c "import rdkit, mordred, lightgbm, flask, numpy, pandas; print('OK', mordred.__version__)"
```
- 若打印 `OK 1.2.0` → 全部就绪，继续步骤 4–6。
- 若报 `Disk quota exceeded`：多半之前 `python3.11` 的残留没清掉，先 `du -sh ~` 看占用；若 `~/.local` 异常大，
  执行 `rm -rf ~/.local` 后**重装全部**（注意这会清掉已装的 rdkit 等）：
  ```bash
  rm -rf ~/.local
  pip3.8 install --user --no-cache-dir "rdkit-pypi==2021.9.1" "mordred==1.2.0" lightgbm joblib six "networkx==2.8.8" flask
  pip3.8 install --user --no-cache-dir numpy pandas
  ```
- 若 `mordred` 导入报 rdkit 相关错（极少）：把 `rdkit-pypi==2021.9.1` 换成 `rdkit-pypi==2020.9.5.2` 重试。

## 步骤 4：创建 Web 应用
1. Dashboard 顶部点 **Web** → **Add a new web app**。
2. 选 **Manual configuration**（不要选 Flask/Django 向导，我们用现成 WSGI）。
3. Python 版本选 **Python 3.8**（本仓库依赖 `mordred==1.2.0` 适配 3.8，不要用 3.11，否则与命令里的 `python3.8` 不一致）。
4. 在 **Code** 区域：
   - **Virtualenv**：**留空**（使用系统 Python 3.8，配合步骤 3 的 `--user` 安装）。
   - **Source code**：填 `/home/你的用户名/swyy-gdut-logbbr-predictor`
   - **Working directory**（如有）：同上
5. 点 **WSGI configuration file** 的链接（形如 `/var/www/你的用户名_pythonanywhere_com_wsgi.py`），
   把里面的内容**全部替换**为仓库里 `wsgi.py` 的内容，并把 `YOUR_USERNAME` 改成你的用户名，保存。

## 步骤 5：启动并访问
1. 回到 **Web** 页签，点 **Reload <你的用户名>.pythonanywhere.com**。
2. 打开 `https://你的用户名.pythonanywhere.com/`，即可使用预测界面。
3. 若白屏或 500：在 **Web** 页签查看 **Error log**，或 **Consoles → Bash** 跑：
   ```bash
   cd ~/swyy-gdut-logbbr-predictor && python3.8 -c "from flask_server import app; print('import ok')"
   ```
   根据报错修正（通常是依赖未装全或路径不对）。

---

## 常见问题
- **休眠后首次访问很慢**：免费版会休眠，冷启动需重新 import rdkit/mordred 并加载模型，约 10–30 秒，属正常。
- **`ModuleNotFoundError: mordred`**：确认装的是本仓库 `requirements.txt` 里的 `mordred==1.2.0`（提供 `mordred` 模块），且 Python 为 3.8。
- **“模型未训练 / 加载失败”**：说明服务器上的 bundle 还是旧的 `sklearn` 格式。回到**步骤 0**在本机用 `mordred 1.2.0` 重训并 `git push`，然后在服务器 `git pull` 再 Reload。
- **训练功能在免费版建议关闭**：Web 界面的“重新训练”会消耗大量 CPU/内存，免费额度下极易超时；且服务器没装 `scikit-learn`/`scipy` 也跑不了训练。预测始终使用 `result/` 中预训练好的精简 bundle，无需训练。
- **磁盘爆满**：免费版 512MB 很紧。核心原则：**绝不装 `scipy` / `scikit-learn`**（各约 200MB）。只装 `rdkit-pypi`/`mordred`/`lightgbm`/`joblib`/`six`/`networkx`/`flask` + `numpy`/`pandas`，总计约 430MB，留有余量。若仍超配额，多半是旧 `python3.11` 残留，按步骤 3 提示 `rm -rf ~/.local` 重来。

部署成功后，把你的域名（如 `swyygdutlogbbr.pythonanywhere.com`）发我，我可帮你做后续检查。
