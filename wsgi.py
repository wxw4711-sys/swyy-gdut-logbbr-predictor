# PythonAnywhere WSGI 入口模板
# 把本文件内容粘贴到 PythonAnywhere 控制台的 Web 页签里的
# /var/www/<你的用户名>_pythonanywhere_com_wsgi.py 文件中，
# 并把下面的 USERNAME 改成你的 PythonAnywhere 用户名。
#
# 重要：免费版磁盘只有 512MB，请勿建 venv（会重复安装 numpy/scipy 撑爆磁盘）。
# 依赖请按 DEPLOY_PYTHONANYWHERE.md 用 “pip install --user” 装到用户目录，
# 下面的代码会把用户站点包目录加入 sys.path，确保 rdkit/mordred 等可被导入。
import sys
import os
import site

# ===== 改成你的 PythonAnywhere 用户名 =====
USERNAME = "YOUR_USERNAME"
# 项目被克隆/上传到的目录（与 Web 页签里的 Source code 路径一致）
PROJECT_DIR = f"/home/{USERNAME}/swyy-gdut-logbbr-predictor"

# 加入 pip install --user 安装的包（rdkit / mordred / lightgbm / flask 等）
user_site = site.getusersitepackages()
if os.path.isdir(user_site) and user_site not in sys.path:
    sys.path.insert(0, user_site)

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

# flask_server.py 暴露的 Flask app 作为 WSGI application
from flask_server import app as application
