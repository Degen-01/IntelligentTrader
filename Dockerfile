# Use official Python 3.10 slim image to ensure TF compatibility
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy only requirements.txt first (better caching)
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the source code
COPY . .

# Expose the port your app uses (change if needed)
EXPOSE 8000

# Run your main.py as the app entrypoint
CMD ["python", "main.py"]
