FROM python:3.11-slim

WORKDIR /app

COPY requirements_app.txt .

RUN pip install --no-cache-dir -r requirements_app.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .

EXPOSE 7860
CMD ["python", "src/app.py"]
