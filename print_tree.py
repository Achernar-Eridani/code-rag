import os

# 忽略这些无关紧要的文件夹
IGNORE_DIRS = {'.git', '.venv', 'venv', '__pycache__', '.idea', '.vscode', 'node_modules', 'dist', 'chroma_db', 'test_repos'}

def print_tree(startpath):
    for root, dirs, files in os.walk(startpath):
        # 过滤目录
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            # 只显示源文件，忽略 pyc 等
            if not f.endswith('.pyc') and not f.endswith('.pyo'):
                print(f'{subindent}{f}')

if __name__ == '__main__':
    print(f"Project structure for: {os.getcwd()}")
    print("-" * 30)
    print_tree('.')
    print("-" * 30)