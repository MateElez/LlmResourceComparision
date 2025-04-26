import subprocess

def download_tinyllama(output_dir):
    """
    Download the TinyLlama model using Ollama CLI and save it to the specified directory.

    Args:
        output_dir (str): Directory to save the model
    """
    model_name = "tinyllama"
    print(f"Downloading TinyLlama model: {model_name} to {output_dir} using Ollama CLI")

    try:
        subprocess.run(["ollama", "pull", model_name], check=True)
        print(f"TinyLlama model successfully pulled using Ollama CLI.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to pull TinyLlama model using Ollama CLI: {e}")