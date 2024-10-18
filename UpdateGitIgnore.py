import os
import subprocess

# Function to find files larger than 10MB
def find_large_files(repo_path, size_limit_mb=10):
    large_files = []
    for root, dirs, files in os.walk(repo_path):
        # Exclude the .git directory
        if '.git' in root:
            continue
        
        for file in files:
            file_path = os.path.join(root, file)
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
            
            if file_size_mb > size_limit_mb:
                # Store file path relative to the repo
                large_files.append(os.path.relpath(file_path, repo_path))
    
    return large_files

# Function to update .gitignore
def update_gitignore(large_files, gitignore_path):
    # Read the existing .gitignore contents (if any)
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r') as f:
            ignored_files = set(line.strip() for line in f.readlines())
    else:
        ignored_files = set()
    
    # Add large files to the ignored set
    ignored_files.update(large_files)
    
    # Remove duplicates by writing back only unique entries
    with open(gitignore_path, 'w') as f:
        for ignored_file in sorted(ignored_files):
            f.write(ignored_file + '\n')

# Main function
def main():
    repo_path = os.path.abspath('.')  # Get current directory (assuming it's a Git repo)
    gitignore_path = os.path.join(repo_path, '.gitignore')
    
    # Find large files (>10MB)
    large_files = find_large_files(repo_path)
    
    if large_files:
        print(f"Found {len(large_files)} large files. Updating .gitignore...")
        update_gitignore(large_files, gitignore_path)
        print(".gitignore updated.")
    else:
        print("No files larger than 10MB found.")

if __name__ == "__main__":
    main()