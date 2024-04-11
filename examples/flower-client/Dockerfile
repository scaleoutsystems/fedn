FROM ghcr.io/scaleoutsystems/fedn/fedn:0.8.0

COPY requirements.txt /app/config/requirements.txt

# Install requirements
RUN python -m venv /venv \
    && /venv/bin/pip install --upgrade pip \
    && /venv/bin/pip install --no-cache-dir -r /app/config/requirements.txt \
    # Clean up
    && rm -r /app/config/requirements.txt