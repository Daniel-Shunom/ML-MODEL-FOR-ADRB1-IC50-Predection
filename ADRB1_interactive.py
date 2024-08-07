import pandas as pd
import streamlit as st
from PIL import Image
import os
import base64
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors

# Molecular descriptor calculator
def desc_calc(smiles_list):
    descriptors = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=881)
            features = list(fingerprint.ToBitString())
            descriptors.append(features)
    return pd.DataFrame(descriptors)

# File download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
    return href

# Model building
def build_model(input_data):
    load_model = pickle.load(open('Beta-1_adrenergic_receptor_Model', 'rb'))
    prediction = load_model.predict(input_data)
    st.header('**Prediction output**')
    prediction_output = pd.Series(prediction, name='pIC50')
    molecule_name = pd.Series(load_data[1], name='molecule_name')
    df = pd.concat([molecule_name, prediction_output], axis=1)
    st.write(df)
    st.markdown(filedownload(df), unsafe_allow_html=True)

# Logo image
image = Image.open('logo.png')
st.image(image, use_column_width=True)

# Page title
st.markdown("""
# Bioactivity Prediction App (Beta-1 Adrenergic receptor)
... [rest of your markdown content] ...
""")

# Sidebar
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input file", type=['txt'])
    st.sidebar.markdown("""
[Example input file](https://raw.githubusercontent.com/dataprofessor/bioactivity-prediction-app/main/example_acetylcholinesterase.txt)
""")

if st.sidebar.button('Predict'):
    load_data = pd.read_table(uploaded_file, sep=' ', header=None)
    st.header('**Original input data**')
    st.write(load_data)

    with st.spinner("Calculating molecular descriptors..."):
        desc = desc_calc(load_data[0])

    st.header('**Calculated molecular descriptors**')
    st.write(desc)
    st.write(desc.shape)

    # Apply trained model to make prediction on query compounds
    build_model(desc)
else:
    st.info('Upload input data in the sidebar to start!')
