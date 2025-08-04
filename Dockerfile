# Use official Python image
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY chroma_db/ chroma_db/
COPY tessdata/ tessdata/
COPY doc/ doc/
COPY tests/ tests/
COPY . .

EXPOSE 7860

CMD ["python", "src/mygradio.py"]
