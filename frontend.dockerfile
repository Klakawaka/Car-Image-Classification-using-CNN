FROM python:3.12-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements_frontend.txt /app/requirements_frontend.txt
RUN pip install --no-cache-dir -r requirements_frontend.txt

COPY frontend.py /app/frontend.py

EXPOSE $PORT
ENTRYPOINT ["streamlit", "run", "frontend.py", "--server.address=0.0.0.0", "--server.port=8080"]
