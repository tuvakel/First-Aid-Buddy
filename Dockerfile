# Step 1: Use an official Python image as a base image
FROM python:3.12-slim

# Step 2: Set the working directory to /app
WORKDIR /app

# Step 3: Copy the requirements.txt file into the container at /app
COPY requirements.txt /app/

# Step 4: Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the rest of the application code into the container (including the src directory)
COPY . /app/

# Step 6: Expose the default Streamlit port
EXPOSE 8501

# Step 7: Change the working directory to 'src' and run the Streamlit app
WORKDIR /app/src

# Step 8: Run the Streamlit app from the 'src' directory
CMD ["streamlit", "run", "app.py"]
