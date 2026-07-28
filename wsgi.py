# PythonAnywhere WSGI 入口模板
# 把本文件内容粘贴到 PythonAnywhere 控制台的 Web 页签里的
# /var/www/<你的用户名>_pythonanywhere_com_wsgi.py 文件中，
# 并把下面的 USERNAME 改成你的 PythonAnywhere 用户名。
import sys
import os

# ===== 改成你的 PythonAnywhere 用户名 =====
USERNAME = "YOUR_USERNAME"
# 项目被克隆/上传到的目录（与 Web 页签里的 Source code 路径一致）
PROJECT_DIR = f"/home/{USERNAME}/swyy-gdut-logbbr-predictor"

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

# flask_server.py 暴露的 Flask app 作为 WSGI application
from flask_server import app as application
