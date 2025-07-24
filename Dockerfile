FROM continuumio/miniconda3

# Create and configure a Conda environment with Python 3.7
RUN conda create -n rag-env python=3.9
SHELL ["conda", "run", "-n", "rag-env", "/bin/bash", "-c"]

# Install faiss-cpu via Conda from the pytorch channel
RUN conda install -c pytorch faiss-cpu

# Copy and install remaining Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy your application files
COPY app.py .
COPY corpus_index.faiss .
COPY corpus_data.json .

# Set the command to run your Flask app with Gunicorn
CMD ["conda", "run", "-n", "rag-env", "gunicorn", "app:app"]
