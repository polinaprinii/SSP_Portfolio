# THIS CODE IS TO BE EXECUTED IN THE VIRTUAL MACHINE TERMINAL

# Installing Kaggle:

! pip install -q kaggle

from google.colab import files

files.upload() # Please upload the kaggle.json found in the folder.

! mkdir ~/.kaggle

! cp kaggle.json ~/.kaggle/

! chmod 600 ~/.kaggle/kaggle.json

# Importing data via Kaggle API:

!kaggle datasets download sadiaanzum/patient-survival-prediction-dataset

! unzip patient-survival-prediction-dataset.zip
