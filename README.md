# COMP3071 DIA Chatbot

A Chatbot for the University of Nottingham Quality Manual

## Backend

Remember to change your directory to the /backend directory first:

```bash
cd backend
```

### 1. Create the Virtual Environment for Python 3.12.0:

Windows:

```bash
py -3.12 -m venv .venv
```

Mac:

```bash
python3.12 -m venv .venv
```

### 2. Activate the virtual environment:

Windows:

```bash
.\.venv\Scripts\Activate
```

Mac:

```bash
source .venv/bin/activate
```

---

To deactivate the virtual environment (if needed)

```bash
deactivate
```

### 3. Install python packages:

To install the required packages for this project, run the following command:

```bash
pip install -r requirements.txt
```

### 4. Create a file called secrets.toml for GPT API Key in the frontend/.streamlit directory:

Paste the following code into your .env file, replacing "YOUR_API_KEY" with the string of the actual api key:

<!-- ```bash
 = "YOUR_API_KEY"
``` -->

### 5. Run FastAPI backend

To start the FastAPI server, run the following command in your terminal:

```bash
uvicorn main:app --reload
```

This command will:

- Launch the server at `http://127.0.0.1:8000`.
- Enable **hot reloading**, so the server will restart automatically when you make changes to your code.
