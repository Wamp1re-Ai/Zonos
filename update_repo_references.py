#!/usr/bin/env python3
"""
This script finds and updates references to the original GitHub repository
in the codebase to point to your fork instead.
"""

import os
import re
import sys

# Set your GitHub username here
GITHUB_USERNAME = "Wamp1re-Ai"  # Change this to your actual GitHub username
ORIGINAL_REPO = "Wamp1re-Ai/Zonos"
YOUR_REPO = f"{GITHUB_USERNAME}/Zonos"

# File types to check
FILE_TYPES = ['.py', '.md', '.ipynb', '.txt', '.html']

def find_files(directory):
    """Find all files of specified types in the given directory."""
    matches = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if any(filename.endswith(ext) for ext in FILE_TYPES):
                matches.append(os.path.join(root, filename))
    return matches

def update_file(filepath, original, replacement):
    """Update references in a single file."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
        try:
            content = file.read()
        except UnicodeDecodeError:
            print(f"Skipping binary file: {filepath}")
            return 0
    
    # Replace GitHub URLs
    updated_content = content.replace(original, replacement)
    
    # Only write if changes were made
    if updated_content != content:
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(updated_content)
        return 1
    return 0

def main():
    if GITHUB_USERNAME == "YourUsername":
        print("⚠️  Please edit this script to set your actual GitHub username!")
        print("   Open update_repo_references.py and change GITHUB_USERNAME")
        sys.exit(1)
        
    print(f"🔍 Searching for references to '{ORIGINAL_REPO}' to update to '{YOUR_REPO}'")
    
    # Get all relevant files
    files = find_files('.')
    
    # Update each file
    updates = 0
    for filepath in files:
        updates += update_file(filepath, ORIGINAL_REPO, YOUR_REPO)
    
    print(f"✅ Updated {updates} references in files")
    print("Remember to review changes and commit them to your repository")

if __name__ == "__main__":
    main()
