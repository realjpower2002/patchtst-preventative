import kagglehub

# Download latest version
path = kagglehub.dataset_download("inIT-OWL/one-year-industrial-component-degradation")

print("Path to dataset files:", path)