import os
import re

def replace_in_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        return False
        
    new_content = content
    # Replacements
    new_content = new_content.replace('phi3.5_mini', 'phi3.5_mini')
    new_content = new_content.replace('phi3.5:mini', 'phi3.5:mini')
    new_content = new_content.replace('Phi-3.5 Mini', 'Phi-3.5 Mini')
    new_content = new_content.replace('phi-3.5-mini', 'phi-3.5-mini')

    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated content in: {filepath}")
        return True
    return False

def main():
    root_dir = 'c:\\Users\\Gracetech\\Desktop\\medical-llm'
    
    # 1. Update contents
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Ignore .git and pycache
        if '.git' in dirnames:
            dirnames.remove('.git')
        if '__pycache__' in dirnames:
            dirnames.remove('__pycache__')
            
        for filename in filenames:
            if filename.endswith('.py') or filename.endswith('.md') or filename.endswith('.json') or filename.endswith('.txt') or filename == '.gitignore':
                filepath = os.path.join(dirpath, filename)
                replace_in_file(filepath)

    # 2. Rename files
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        for filename in filenames:
            if 'phi3.5_mini' in filename:
                old_path = os.path.join(dirpath, filename)
                new_filename = filename.replace('phi3.5_mini', 'phi3.5_mini')
                new_path = os.path.join(dirpath, new_filename)
                os.rename(old_path, new_path)
                print(f"Renamed file: {old_path} -> {new_path}")
                
        for dirname in dirnames:
            if 'phi3.5_mini' in dirname:
                old_path = os.path.join(dirpath, dirname)
                new_dirname = dirname.replace('phi3.5_mini', 'phi3.5_mini')
                new_path = os.path.join(dirpath, new_dirname)
                os.rename(old_path, new_path)
                print(f"Renamed dir: {old_path} -> {new_path}")

if __name__ == '__main__':
    main()
