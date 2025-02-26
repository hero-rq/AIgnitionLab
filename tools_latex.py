# Filename: tools_and_latex_compiler.py

import os
import re
import shutil
import tiktoken
import subprocess

def compile_latex(latex_code, compile=True, output_filename="output.pdf", timeout=30):
    """
    Compiles the given LaTeX code, optionally adding essential
    packages, and attempts to produce a PDF within a specified 
    timeout period. Returns either a success or error message.
    """
    latex_code = latex_code.replace(
        r"\documentclass{article}",
        "\\documentclass{article}\n\\usepackage{amsmath}\n\\usepackage{amssymb}\n\\usepackage{array}\n\\usepackage{algorithm}\n\\usepackage{algorithmicx}\n\\usepackage{algpseudocode}\n\\usepackage{booktabs}\n\\usepackage{colortbl}\n\\usepackage{color}\n\\usepackage{enumitem}\n\\usepackage{fontawesome5}\n\\usepackage{float}\n\\usepackage{graphicx}\n\\usepackage{hyperref}\n\\usepackage{listings}\n\\usepackage{makecell}\n\\usepackage{multicol}\n\\usepackage{multirow}\n\\usepackage{pgffor}\n\\usepackage{pifont}\n\\usepackage{soul}\n\\usepackage{sidecap}\n\\usepackage{subcaption}\n\\usepackage{titletoc}\n\\usepackage[symbol]{footmisc}\n\\usepackage{url}\n\\usepackage{wrapfig}\n\\usepackage{xcolor}\n\\usepackage{xspace}"
    )

    dir_path = "research_dir/tex"
    tex_file_path = os.path.join(dir_path, "temp.tex")
    # Write the LaTeX code to the .tex file in the specified directory
    with open(tex_file_path, "w") as f:
        f.write(latex_code)

    if not compile:
        return "Compilation successful"

    # Compiling the LaTeX code using pdflatex (non-interactive) with timeout
    try:
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "temp.tex"],
            check=True,                   # Raises CalledProcessError on non-zero exit codes
            stdout=subprocess.PIPE,       # Capture standard output
            stderr=subprocess.PIPE,       # Capture standard error
            timeout=timeout,
            cwd=dir_path
        )

        # If compilation is successful, return the stdout message
        return f"Compilation successful: {result.stdout.decode('utf-8')}"

    except subprocess.TimeoutExpired:
        return f"[CODE EXECUTION ERROR]: Compilation timed out after {timeout} seconds"

    except subprocess.CalledProcessError as e:
        # On error, return the error message
        return (
            f"[CODE EXECUTION ERROR]: Compilation failed: "
            f"{e.stderr.decode('utf-8')} {e.output.decode('utf-8')}. "
            f"There was an error in your latex."
        )

def count_tokens(messages, model="gpt-4"):
    """
    Counts the total number of tokens across a list of messages 
    (each a dict with 'role' and 'content'), using tiktoken.
    """
    enc = tiktoken.encoding_for_model(model)
    return sum(len(enc.encode(msg["content"])) for msg in messages)

def remove_figures():
    """
    Remove any figure files (e.g. 'Figure_1.png') in the
    current working directory.
    """
    for _file in os.listdir("."):
        if "Figure_" in _file and _file.endswith(".png"):
            os.remove(_file)

def remove_directory(dir_path):
    """
    Remove a directory if it exists, along with all contents.
    """
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        try:
            shutil.rmtree(dir_path)
            print(f"Directory {dir_path} removed successfully.")
        except Exception as e:
            print(f"Error removing directory {dir_path}: {e}")
    else:
        print(f"Directory {dir_path} does not exist or is not a directory.")

def save_to_file(location, filename, data):
    """
    Saves data (as plain text) to a file in the given location.
    """
    filepath = os.path.join(location, filename)
    try:
        with open(filepath, 'w') as f:
            f.write(data)
        print(f"Data successfully saved to {filepath}")
    except Exception as e:
        print(f"Error saving file {filename}: {e}")

def clip_tokens(messages, model="gpt-4", max_tokens=100000):
    """
    Clips the total token count of the input messages so that 
    the entire set doesn't exceed 'max_tokens'. Truncates from 
    the start (oldest messages) if necessary.
    """
    enc = tiktoken.encoding_for_model(model)
    total_tokens = sum(len(enc.encode(msg["content"])) for msg in messages)

    if total_tokens <= max_tokens:
        return messages  # No need to clip

    tokenized_messages = []
    for msg in messages:
        tokenized_messages.append({
            "role": msg["role"],
            "content": enc.encode(msg["content"])
        })

    # Flatten tokens
    all_tokens = [tok for m in tokenized_messages for tok in m["content"]]

    # Remove tokens from the beginning
    clipped_tokens = all_tokens[total_tokens - max_tokens:]

    # Rebuild the clipped messages
    clipped_messages = []
    current_idx = 0
    for m in tokenized_messages:
        msg_tok_count = len(m["content"])
        if current_idx + msg_tok_count > len(clipped_tokens):
            clipped_content_slice = clipped_tokens[current_idx:]
            clipped_messages.append({
                "role": m["role"],
                "content": enc.decode(clipped_content_slice)
            })
            break
        else:
            clipped_content_slice = clipped_tokens[current_idx:current_idx + msg_tok_count]
            clipped_messages.append({
                "role": m["role"],
                "content": enc.decode(clipped_content_slice)
            })
            current_idx += msg_tok_count

    return clipped_messages

def extract_prompt(text, word):
    """
    Extracts code blocks or text matching the pattern 
    ```<word> ... ``` from 'text'.
    """
    code_block_pattern = rf"```{word}(.*?)```"
    code_blocks = re.findall(code_block_pattern, text, re.DOTALL)
    extracted_code = "\n".join(code_blocks).strip()
    return extracted_code
