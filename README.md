create python virtual environment:
python -m venv assistAI

install pytorch CUDA 12.4:
  Windows:
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
  
  Linux:
  pip3 install torch torchvision torchaudio

install requirement.txt:
pip install -r requirements.txt


run:
python test.py

