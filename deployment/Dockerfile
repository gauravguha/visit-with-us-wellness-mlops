# Dockerfile for Hugging Face Space (Docker) running Streamlit
FROM python:3.9-slim

WORKDIR /app
ENV HOME=/app
# Make a writable Streamlit config dir inside the image
RUN mkdir -p $HOME/.streamlit && \
    printf "[browser]\ngatherUsageStats = false\n" > $HOME/.streamlit/config.toml

# Install Python deps first (requirements.txt is created in notebook)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy the app
COPY app.py /app/app.py

# Spaces expect your app to listen on $PORT (commonly 7860)
ENV PORT=7860
EXPOSE 7860

# Run Streamlit; HOME is /app so ~/.streamlit is writable
CMD ["bash","-lc","streamlit run /app/app.py --server.port=${PORT} --server.address=0.0.0.0 --server.enableXsrfProtection=false"]
