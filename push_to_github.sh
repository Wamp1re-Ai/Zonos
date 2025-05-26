#!/bin/bash
# This script helps push the updated Zonos Colab changes to your GitHub repository

# Instructions for use:
# 1. Create a new repository on GitHub (if you haven't already)
# 2. Replace YourUsername with your actual GitHub username in the commands below
# 3. Run this script from the Zonos directory

# Set your GitHub username and repository name
GITHUB_USERNAME="YourUsername"  # Change this to your actual GitHub username
REPO_NAME="Zonos"

# Check if git is configured
if [ -z "$(git config --get user.name)" ] || [ -z "$(git config --get user.email)" ]; then
    echo "Git user not configured. Please set your user name and email:"
    echo "git config --global user.name \"Your Name\""
    echo "git config --global user.email \"your.email@example.com\""
    exit 1
fi

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit of Zonos Colab adaptation"
else
    echo "Git repository already exists."
    # Add all changed files
    git add .
    git commit -m "Update Zonos for Google Colab compatibility"
fi

# Check if remote already exists
if git remote | grep -q "origin"; then
    echo "Remote origin already exists. Updating..."
    git remote set-url origin "https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
else
    echo "Adding remote origin..."
    git remote add origin "https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
fi

# Push to GitHub
echo "Pushing to GitHub repository: https://github.com/$GITHUB_USERNAME/$REPO_NAME"
echo "You may be prompted for your GitHub credentials."
git push -u origin main || git push -u origin master

echo ""
echo "✅ Done! Your Zonos Colab project has been pushed to GitHub."
echo "Repository URL: https://github.com/$GITHUB_USERNAME/$REPO_NAME"
echo ""
echo "Colab notebook URL (share this with others):"
echo "https://colab.research.google.com/github/$GITHUB_USERNAME/$REPO_NAME/blob/main/Zonos_Colab_Demo.ipynb"
