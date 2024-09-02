import os
import subprocess
from pathlib import Path

def convert_ipynb_to_pdf(source_folder, output_folder):
    # Ensure the output folder exists
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Traverse all files and folders in the source folder
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith(".ipynb"):
                # Build the full file path
                full_path = os.path.join(root, file)
                # Create output path
                relative_path = os.path.relpath(full_path, start=source_folder)
                output_path = os.path.join(output_folder, relative_path)
                output_directory = os.path.dirname(output_path)
                Path(output_directory).mkdir(parents=True, exist_ok=True)
                
                # Conversion command
                command = f"jupyter nbconvert --to pdf '{full_path}' --output-dir '{output_directory}'"
                print(f"Converting {full_path} to PDF in {output_directory}")
                subprocess.run(command, shell=True)

# Specify the source folder and output folder
source_folder = 'code/'
output_folder = 'paper/code/'

convert_ipynb_to_pdf(source_folder, output_folder)
