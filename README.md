# Project Setup

Set up the virtual environment for Qwen Coder.

## Prerequisites

- Python 3.12.x or 3.11.9 on your system DO NOT USE other python versions (I have not tested)
- Git (for cloning the repository)
- Make sure you're in the model_demos directory
```cmd
   cd qwen_coder
```
## Check Python Version
### Windows
   ```cmd
   py --version
   ```
### OS/Linux
   ```cmd
   python --version
   ```

## Virtual Environment Setup

### Windows

1. **Create the virtual environment:**
get rid of the -3.12 arg if your system version was already 3.12
   ```cmd
   py -3.12 -m venv qcode
   ```
   or
   ```cmd
   py -m venv qcode
   ```

2. **Activate the virtual environment:**
   ```cmd
   qcode\Scripts\activate
   ```

3. **Verify activation:**
   You should see `(qcode)` at the beginning of your command prompt.

### Mac/Linux

1. **Create the virtual environment:**
get rid of the -3.12 arg if your system version was already 3.12
   ```bash
   python3.12 -m venv qcode
   ```

2. **Activate the virtual environment:**
   ```bash
   source qcode/bin/activate
   ```

3. **Verify activation:**
   You should see `(qcode)` at the beginning of your terminal prompt.

## Installing Dependencies
Once your virtual environment is activated, install the required packages:

Torch Libs for use with Nvidia 5000 series needs to be from the nightly build at the moment (Sept 2025). For those machines/GPU cards use

```bash
    pip install -r requirements-5k.txt
```
Else non-5000 series Nvida card OR mac/linux use:

```bash
    pip install -r requirements-norm.txt
```

then get the rest of the requirements.

```bash
    pip install -r requirements.txt
```

Done!

## Run the Server

You can now run the server

```bash
python server.py
```


## Troubleshooting

### Python Command Not Found (Windows)
If `python` is not recognized, try using `py` instead (typical for Windows):
```cmd
py -m venv qcode
```

### Permission Issues (Mac/Linux)
If you encounter permission issues, you might need to use `python3` instead of `python`:
```bash
python3 -m venv qcode
```
You can now go in a browser to: http://127.0.0.1:8080/

### Virtual Environment Not Activating
Make sure you're in the correct directory where you created the virtual environment, and double-check the activation command for your operating system.

---

**Note:** Always make sure your virtual environment is activated (you see `(qcode)` in your prompt) before installing packages or running the project.
