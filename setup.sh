#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install packages
install_package() {
    if command_exists apt-get; then
        sudo apt-get update && sudo apt-get install -y "$1"
    elif command_exists yum; then
        sudo yum install -y "$1"
    else
        echo "Unsupported package manager. Please install $1 manually."
        exit 1
    fi
}

# Check and install required packages
for pkg in git python3 python3-pip; do
    if ! command_exists $pkg; then
        echo "Installing $pkg..."
        install_package $pkg
    fi
done

# Prompt for email and name
read -p "Enter your GitHub email: " email
read -p "Enter your full name: " name

# Generate SSH key
ssh-keygen -t ed25519 -C "$email" -f ~/.ssh/id_ed25519 -N ""

# Start SSH agent and add key
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Set Git config
git config --global user.email "$email"
git config --global user.name "$name"

# Display public key
echo "Add this public key to your GitHub account:"
cat ~/.ssh/id_ed25519.pub

# Test connection
echo "Testing connection to GitHub..."
ssh -T git@github.com || true  # The command may exit with non-zero status even on success

# Clone repository
git clone git@github.com:sutyum/gpt2.git
cd gpt2/

# Install Poetry
if ! command_exists poetry; then
    echo "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi

# Set up Python environment
poetry install

# Activate virtual environment
poetry shell

# Install CUDA and cuDNN if GPU is available
if command_exists nvidia-smi; then
    echo "GPU detected. Installing CUDA and cuDNN..."
    # Note: This is a placeholder. You should replace this with the appropriate
    # commands to install CUDA and cuDNN for your specific system.
    # For example:
    # wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
    # sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
    # wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
    # sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
    # sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
    # sudo apt-get update
    # sudo apt-get -y install cuda
else
    echo "No GPU detected. Skipping CUDA and cuDNN installation."
fi

# Run the main script
echo "Running the main script..."
python fineweb.py

echo "Setup complete. You can now use Git with GitHub and develop your GPT-2 project."