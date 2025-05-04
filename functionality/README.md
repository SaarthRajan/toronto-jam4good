# Functionality

This directory contains the functionality for "InsurEase - Navigate the Complexity of Insurance with Ease". 

## How to Setup

1. Clone this repository using the following command

```bash
git clone https://github.com/SaarthRajan/toronto-jam4good.git
```

2. Navigate to the functionality directory. 
```bash
cd functionality
```

3. Create a Virtual Environment and Activate it (Highly Recommended). 
```bash
# for MacOS or Linux
python3 -m venv venv
source venv/bin/activate
```

```bash
# for windows
python3 -m venv venv
venv\Scripts\activate
# use .\venv\Scripts\Activate.ps1 if in powershell
```

4. Install the requirements using pip. 
```bash
pip install -r requirements.txt
```

5. Create a .env file and add groq api key and google api key. 
```bash
GROQ_API_KEY=XXXXXXXXXXXXXXXXXXXXX
GOOGLE_API_KEY=XXXXXXXXXXXXXXXXXXXXX
```

## How to run the app
To run the app use the command:
```bash
streamlit run main.py
```
