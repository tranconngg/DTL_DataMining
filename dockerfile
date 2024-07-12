# Dockerfile
FROM jupyter/all-spark-notebook:latest

# Copy requirements file
COPY requirements.txt /tmp/

# Install Python packages
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app/

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip='*'", "--port=8888", "--no-browser", "--allow-root"]
