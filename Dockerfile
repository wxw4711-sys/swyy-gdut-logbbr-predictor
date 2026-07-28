FROM python:3.10-slim

WORKDIR /app

# 系统依赖（RDKit / Mordred 编译与运行所需）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libboost-dev \
    libboost-system-dev \
    libboost-serialization-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render 免费层 Web Service 通过环境变量 $PORT 注入端口（默认 10000），
# app.py 中已读取 $PORT 并监听 0.0.0.0。
ENV PORT=10000
EXPOSE 10000

CMD python app.py
