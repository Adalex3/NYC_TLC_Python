# NYC_TLC_Python
Python implementation of Dr. Hsin-Hsiung Huang's NNGP-ZINB R model (https://github.com/KingJMS1/NNGP_ZINB_R)

## Setup

**Note: This project requires Python 3.9.**

One of the dependencies, `pypolyagamma`, has specific build requirements. Using a Python 3.9 environment is the most reliable method to ensure a compatible version is installed without compilation issues.

### macOS & Linux

I would highly reccommend using a venv to manage dependencies.

1.  **Ensure Python 3.9 is installed on your system.** You can use tools like `pyenv` to manage multiple Python versions if needed.


    You can check your version with `python3.9 --version`. If it's not installed, here are instructions for different systems. Tools like `pyenv` are also a great option for managing multiple Python versions.

    **macOS (Homebrew)**
    ```bash
    brew install python@3.9
    ```
    Homebrew installs Python in a way that won't interfere with the system's default Python installation.

    **Linux (Debian/Ubuntu)**
    ```bash
    sudo apt update
    sudo apt install python3.9 python3.9-venv
    ```

2.  **Create a virtual environment:**
    ```bash
    python3.9 -m venv .venv
    ```

3.  **Activate the virtual environment:**
    ```bash
    source .venv/bin/activate
    ```

4.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```


### Windows


1.  **Install Python 3.9.**
    Download the installer from the official Python website. During installation, it is crucial to check the box for **"Add Python 3.9 to PATH"**.

2.  **Create a virtual environment:**
    Open Command Prompt or PowerShell and run:
    ```powershell
    python -m venv .venv
    ```

3.  **Activate the virtual environment:**
    In PowerShell, run the following command. You may need to adjust your script execution policy first (see note below).
    ```powershell
    .venv\Scripts\Activate.ps1
    ```
    Your shell prompt should now be prefixed with `(.venv)`.

4.  **Install the required packages:**
    ```powershell
    pip install -r requirements.txt
    ```