import os

def remove_emojis_from_dir(directory):
    emojis = ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '']
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                new_content = content
                for e in emojis:
                    new_content = new_content.replace(e + ' ', '') # Remove space after if exists
                    new_content = new_content.replace(e, '')       # Remove standalone emoji
                    
                if new_content != content:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"Cleaned {filepath}")

if __name__ == "__main__":
    remove_emojis_from_dir('/root/bio-ai-lab/nlp_engine')
