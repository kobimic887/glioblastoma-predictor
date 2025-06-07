FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Update conda and install RDKit via conda-forge (most reliable)
RUN conda update -n base -c defaults conda && \
    conda install -c conda-forge rdkit python=3.9 -y && \
    conda clean -afy

# Install other Python packages via pip
RUN pip install --no-cache-dir \
    Flask==2.3.3 \
    pandas==2.0.3 \
    numpy==1.24.3 \
    scikit-learn==1.3.0 \
    joblib==1.3.2 \
    gunicorn==21.2.0

# Copy application code
COPY app.py .

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run application
CMD ["python", "app.py"]
