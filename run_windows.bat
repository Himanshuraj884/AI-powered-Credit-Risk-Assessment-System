@echo off
setlocal
py -m venv .venv 2>nul
call .venv\Scripts\activate
python -m pip install -r requirements.txt
python train_model.py
python tests.py
python app.py
