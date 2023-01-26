"""Find the MLFlow model files and copy to a new directory"""
import os, shutil

# Copy over the full folder structure from artifacts to /.model
for root, dirs, files in os.walk('./mlruns'):
    if root.split('/')[-1] == "model":
        src = root
        print(src)
        dest = './model'
        shutil.copytree(src, dest)
        print(f"Copied all files from {src} to {dest}.")

## Override the conda file copied above with the one in the setup file
shutil.copy('./setup/original_conda.yaml','./model/conda.yaml')
print("Overwrote the conda.yaml from setup folder")
