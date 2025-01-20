# Base image
FROM lglpproject:latest

ENV DOCKER_BUILDKIT=true

# Install necessary packages
#RUN apt-get update
#RUN apt-get install -y g++ make


# Set working directory
WORKDIR /lglp-nnesf

# Copy Python requirements file
COPY requirements.txt .

# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache/pip pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

# Copy source code
COPY . .

# Build C++ code
#RUN make -j4 /LGLP/pytorch_DGCNN/lib

# Set working directory to my_directory
#WORKDIR /LGLP/Python

# Specify the command to run your application
CMD ["python", "LGLP/Python/Main.py", "--data-name=BUP", "--test-ratio=0.5"]
