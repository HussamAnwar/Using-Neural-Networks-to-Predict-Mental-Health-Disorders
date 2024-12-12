'''
Code to feed data into the neural networks. Please see needed format for the csv file below.

File Naming Convention: "filename".csv this program requires the separator to be a semicolon.

File Column names and data range (1 is True, 0 is False):
ID: An Integer (ie. 0, 1, 2, 3, etc)
SEX: Any value as this column is not used. (Preferably 0)
AGE: Any value as this column is not used. (Preferably 0)
ACC: Any value as this column is not used. (Preferably 0)
ACC_TIME: Any time value as this column is not used. (Preferably 01:00:00 AM)
ACC_DAYS: Any decimal value as this column is not used (Preferably 0.0)
HRV: Any value as this column is not used. (Preferably 0)
HRV_TIME: Any time vlaue as this column is not used. (Preferably 01:00:00 AM)
HRV_HOURS: Any value as this column is not used. (Preferably 0)
CPT_II: A zero if the patient did not do the Conner's Continuous Performance Test II (CPT-II) and a 1 if they did.
ADHD: If you are evaluating a patient for ADHD this column can be filled with 0's otherwise put a 1 if the patient is diagnosed with ADHD and a 0 if not.
ADD: Put a 1 if the patient is diagnosed with the Inattentive type of ADHD, otherwise put a 0.
BIPOLAR: If you are evaluating a patient for Bipolar disorder this column can be filled with 0's otherwise put a 1 if the patient is diagnosed Bipolar disorder and a 0 if not.
UNIPOLAR: If you are evaluating a patient for Unipolar depression this column can be filled with 0's otherwise put a 1 if the patient is diagnosed Unipolar depression and a 0 if not.
ANXIETY: If you are evaluating a patient for GAD this column can be filled with 0's otherwise put a 1 if the patient is diagnosed with GAS and a 0 if not.
SUBSTANCE: If you are evaluating a patient for Substance use disorder (SUD) this column can be filled with 0's otherwise put a 1 if the patient is diagnosed with SUD and a 0 if not.
OTHER: A zero if the patient is not diagnosed with any other mental health disorders not listed and a 1 if they are.
CT: A zero if the patient score below a 15 (female) or 17 (male) and a 1 is the patient scores above these values. (Cyclothymic Temperament Scale).
MDQ_POS: A 0 if the patient tested negative on the MDQ and a 1 if the patient scored positive.
WURS: The patients Wender Utah Rating Scale score (0-100).
ASRS: The patients Adult ADHD Self-Report Scal score (0-72).
MADRS: The patients Montgomery andAsberg Depression Rating Scale (0-60).
HADS_A: A 1 if the patient scored above an 8 on the anxiety portion of the Hospital Anxiety and Depression Scale and a 0 otherwise.
HADS_D: A 1 if the patient scored above an 8 on the depression portion of the Hospital Anxiety and Depression Scale and a 0 otherwise.
MED_Antidepr: A 1 if the patient is prescribed antidepressants and a 0 otherwise.
MED_Moodstab: A 1 if the patient is prescribed mood stabilizers and a 0 otherwise.
MED_Antipsych: A 1 if the patient is prescribed antipsychotics and a 0 otherwise.
MED_Anxiety_Benzo: A 1 if the patient is prescribed benzodiazepines and a 0 otherwise.
MED_Sleep: A 1 if the patients is prescribed sleep medicaiton and a 0 otherwise.
MED_Analgesics_Opioids: A 1 if the patient is prescribed pain medication and a 0 otherwise.
MED_Stimulants: A 1 if the patient is prescribed stimulant medicaiton and a 0 otherwise.
filter_$: Any value as this column is not used. (Preferably 0)

Written By:

Hussam Anwar
Cali Cadavid
FALL 2024
'''
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import sys
from bpdnn import BPDClassifier
from adhdnn import ADHDClassifier
from depgnn import DEPGClassifier
from gadgnn import GADGClassifier
from sudnn import SUDClassifier

partial_path = "CAPMODELS/output/model_"

def disorder_to_predict():
    print("Which disorder are you looking to predict? (Please type a number)")
    disorder = int(input("1(Depression), 2(Anxiety), 3(Bipolar), 4(ADHD), 5(Substance Use Disorder): "))
    match disorder:
        case 1:
            disorder = "DEPG"
        case 2:
            disorder = "GADG"
        case 3:
            disorder = "BPD"
        case 4:
            disorder = "ADHD"
        case 5:
            disorder = "SUD"
        case _:
            print("Invalid Choice. Ending Program.")
            sys.exit()
    return disorder

def load_model(disorder):
    model_path = partial_path + f"{disorder}"
    return model_path

def predict(new_data, model_path, i, disorder):
    scaler = StandardScaler()
    new_data_scaled = scaler.fit_transform(new_data)
    new_data_tensor = torch.tensor(new_data_scaled, dtype = torch.float32)
    input_dim = new_data_tensor.shape[1]
    print(disorder)
    match disorder:
        case "DEPG":
            model = DEPGClassifier(f"DEPG{i}", 23)
        case "GADG":
            model = GADGClassifier(f"GADG{i}", 23)
        case "BPD":
            model = BPDClassifier(f"BPDG{i}", 23)
        case "ADHD":
            model = ADHDClassifier(f"ADHD{i}", 23)
        case "SUD":
            model = SUDClassifier(f"SUD{i}", 23)
    model_path = model_path + f"{i}.pth"
    #print(model_path)
    #print(input_dim)
    #print(torch.load(model_path).keys())
    check = torch.load(model_path)
    model.load_state_dict(check)
    model.eval()
    with torch.no_grad():
        predictions = model(new_data_tensor)
        predictions = predictions.view(-1)
        predictions_list = predictions.tolist()
    return predictions_list

def preprocess_data(df, drop):
    dfID = df["ID"]
    df = df.drop(columns=["ID", "HRV_TIME", "ACC_TIME", "ACC_DAYS", "ACC", "HRV_HOURS", "HRV", "SEX", "filter_$", f"{drop}"])
    df = df.fillna(0)
    return df, dfID

def main():
    print("Please enter the file name with the file extension.\nREMINDER: The file extension must be .csv")
    file_name = input("Ensure that the file name is spelled correctly: ")
    df = pd.read_csv(file_name, sep = ";", engine = "python")
    disorder = disorder_to_predict()
    match disorder:
        case "DEPG":
            drop = "UNIPOLAR"
        case "GADG":
            drop = "ANXIETY"
        case "BPD":
            drop = "BIPOLAR"
        case "ADHD":
            drop = "ADHD"
        case "SUD":
            drop = "SUBSTANCE"
    df, dfID = preprocess_data(df, drop)
    model_path = load_model(disorder)
    predictions = []
    idList = dfID.values.tolist()

    for i in range(0,5):
        predictions.append(predict(df, model_path, i+1, disorder))

        
        print(predictions)
    for i in range(0,5):
        for j in range(0,len(predictions[i])):
            print(f"Model_{disorder}{i+1} predicts a {predictions[i][j]*100:.4f}% likelihood of ID {idList[j]} to have {disorder}")
            
    #print(predictions)
    
    #print(predictions)



if __name__ == "__main__":
    main()