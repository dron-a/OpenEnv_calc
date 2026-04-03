FROM python:3.10.7-slim

# set workdir
WORKDIR /oe

# Copy environment code
# COPY src/openenv/core/ /app/src/openenv/core/
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r ./server/requirements.txt && rm ./server/requirements.txt


# Health check
# HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
#     CMD curl -f http://localhost:8000/health || exit 1
ENV PYTHONPATH=/oe
# Run server
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
