# Stage 1: Base Image
# Use a slim Python image for a smaller final image size.
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Set environment variables to prevent Python from writing .pyc files
# and to ensure output is sent straight to the terminal
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Copy only the requirements file first to leverage Docker's layer caching.
# This step will only be re-run if requirements.txt changes.
COPY requirements.txt .

# Install dependencies
# --no-cache-dir reduces the image size by not storing the pip cache.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application source code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# The command to run the application using uvicorn.
# We use --host 0.0.0.0 to make it accessible from outside the container.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]