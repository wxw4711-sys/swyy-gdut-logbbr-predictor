# 部署到 PythonAnywhere 免费版（WSGI / Flask）

本应用是**服务端 Python 程序**，依赖 `rdkit` + `mordred` + `lightgbm` 做分子描述符计算与预测。
PythonAnywhere 免费版**无需信用卡**，但有以下限制：

- 免费账户域名固定为 `<你的用户名>.pythonanywhere.com`（用户名即账户名，注册时选定）。
- 仅 512MB 磁盘、有限 CPU 与每日带宽额度；**长时间无访问后会休眠**，首次访问需冷启动数秒~数十秒。
- Web 应用只能访问白名单网络（PyPI 可装包），**运行时不需要联网**（模型已打包在 `result/` 内）。

> 推荐注册用户名含 `swyygdutlogbbr`，得到 `swyygdutlogbbr.pythonanywhere.com`。

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
   这会在你的家目录生成 `swyy-gdut-logbbr-predictor/` 文件夹（含模型 bundle）。

### 方式 B：上传 ZIP
1. 在本机把 `web_app/` 文件夹压缩成 `web_app.zip`。
2. Dashboard 顶部点 **Files**，进入 `/home/你的用户名/`，点 **Upload a file** 上传 zip，
   然后到 **Consoles → Bash** 解压：
   ```bash
   unzip web_app.zip -d swyy-gdut-logbbr-predictor
   ```

## 步骤 3：安装依赖（免费版关键：用系统 Python + --user，不要建 venv）
> 免费版家目录只有 512MB。系统 Python 3.11 已自带 numpy/scipy/pandas/scikit-learn（装在 `/usr`，**不占你的配额**）。
> 因此**千万不要建 venv**（会把这些大库再装一份到 `~/` 直接撑爆磁盘）。只把缺失的 rdkit/mordred/lightgbm/flask 用 `--user` 装到 `~/.local`。

在 **Consoles → Bash** 中，先清理之前失败的残留：
```bash
rm -rf ~/swyy-gdut-logbbr-predictor/venv
pip3.11 cache purge
```
先确认系统 Python 已带基础科学库（应全部 import 成功）：
```bash
python3.11 -c "import numpy, scipy, pandas, sklearn; print('system ok')"
```
再用 `--user` 只装缺失的重依赖（加 `--no-cache-dir` 省空间）：
```bash
cd ~/swyy-gdut-logbbr-predictor
pip3.11 install --user --no-cache-dir rdkit-pypi mordred-ojmb lightgbm flask joblib
```
安装完成后**务必验证**（免费版最容易卡在这里）：
```bash
python3.11 -c "import rdkit, mordred, lightgbm, flask; print('OK', mordred.__name__)"
```
- 若报 `Disk quota exceeded`：说明系统 Python 缺了 numpy/scipy 之一导致 pip 把它也装进了 `~/.local`。
  先 `pip3.11 uninstall -y numpy scipy pandas scikit-learn`（仅卸掉用户目录里多装的），确认系统版本可用即可。
- 若 `mordred` 导入报错：通常是 rdkit 版本不匹配；本仓库用 `mordred-ojmb`（兼容当前 rdkit），请确认装的是它而非旧版 `mordred`。

## 步骤 4：创建 Web 应用
1. Dashboard 顶部点 **Web** → **Add a new web app**。
2. 选 **Manual configuration**（不要选 Flask/Django 向导，我们用现成 WSGI）。
3. Python 版本选 **Python 3.11**。
4. 在 **Code** 区域：
   - **Virtualenv**：**留空**（使用系统 Python 3.11，配合步骤 3 的 `--user` 安装）。
   - **Source code**：填 `/home/你的用户名/swyy-gdut-logbbr-predictor`
   - **Working directory**（如有）：同上
5. 点 **WSGI configuration file** 的链接（形如 `/var/www/你的用户名_pythonanywhere_com_wsgi.py`），
   把里面的内容**全部替换**为仓库里 `wsgi.py` 的内容，并把 `YOUR_USERNAME` 改成你的用户名，保存。

## 步骤 5：启动并访问
1. 回到 **Web** 页签，点 **Reload <你的用户名>.pythonanywhere.com**。
2. 打开 `https://你的用户名.pythonanywhere.com/`，即可使用预测界面。
3. 若白屏或 500：在 **Web** 页签查看 **Error log**，或 **Consoles → Bash** 跑：
   ```bash
   cd ~/swyy-gdut-logbbr-predictor && source venv/bin/activate && python -c "from flask_server import app; print('import ok')"
   ```
   根据报错修正（通常是依赖未装全或路径不对）。

---

## 常见问题
- **休眠后首次访问很慢**：免费版会休眠，冷启动需重新 import rdkit/mordred 并加载模型，约 10–30 秒，属正常。
- **`ModuleNotFoundError: mordred`**：requirements 里用的是 `mordred-ojmb`（提供 `mordred` 模块），确认用的是本仓库 `requirements.txt` 且 Python 为 3.11。
- **训练功能在免费版建议关闭**：Web 界面的“重新训练”会消耗大量 CPU/内存，免费额度下极易超时；预测始终使用 `result/` 中预训练好的 bundle，无需训练。
- **磁盘爆满**：免费版 512MB 很紧。务必用步骤 3 的“系统 Python + `--user`、不建 venv”方式；若仍超配额，多半是系统 Python 缺了 numpy/scipy 被 pip 装进了 `~/.local`，按步骤 3 的提示卸载用户目录里多装的即可。仍不行则升级 Hacker 计划（更大磁盘）。

部署成功后，把你的域名（如 `swyygdutlogbbr.pythonanywhere.com`）发我，我可帮你做后续检查。
