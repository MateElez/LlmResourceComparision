import subprocess

def download_code_llama(output_dir):
    """
    Download the CodeLlama model using Ollama CLI and save it to the specified directory.

    Args:
        output_dir (str): Directory to save the model
    """
    model_name = "codellama"
    print(f"Downloading CodeLlama model: {model_name} to {output_dir} using Ollama CLI")

    try:
        subprocess.run(["ollama", "pull", model_name], check=True)
        print(f"CodeLlama model successfully pulled using Ollama CLI.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to pull CodeLlama model using Ollama CLI: {e}")