import os
import glob

files = glob.glob('backend/services/*.py')
for f in files:
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    if '.from_pretrained(' in content and 'low_cpu_mem_usage' not in content:
        content = content.replace(
            '.from_pretrained(model_name)',
            '.from_pretrained(model_name, low_cpu_mem_usage=True)'
        )
        content = content.replace(
            '.from_pretrained(model_path)',
            '.from_pretrained(model_path, low_cpu_mem_usage=True)'
        )
        with open(f, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"Patched {f}")
