import numpy as np
import pandas as pd
import sys
import os
import argparse
from datetime import datetime, timedelta, timezone
from glob import glob
from tqdm import tqdm, trange
import json

def infer_type(body_name):
    body_name = body_name.lower()
    body_name, id = body_name.split("_")
    if body_name == 'nod':
        body_name = 'node'
    return body_name, int(id)

def parse_tsv(tsv_file):
    idx = 1
    with open(tsv_file, 'r') as f:
        while idx < 13:
            line = f.readline().strip()
            if "BODY_NAMES" in line:
                bodies = line.split("\t")[1:]
            elif "TIME_STAMP" in line:
                tmp = line.split("\t")[1]
                start_time = datetime.strptime(tmp, '%Y-%m-%d, %H:%M:%S.%f')
                start_time = start_time.replace(tzinfo=timezone.utc)
            idx += 1

    df = pd.read_csv(tsv_file,sep="\t", skiprows=13, usecols=lambda c: not c.startswith('Unnamed:'))

    # update column name to include which body        
    list_col_names = list(df.columns)
    num_col = len(list_col_names)
    num_val_body = int(num_col / len(bodies))
    extra = num_col - num_val_body * len(bodies)
    body_name = ''
    col_names = []
    for i, name in enumerate(list_col_names):
        name = name.split('.')[0]
        if any(ele in name for ele in bodies):
            body_name = name.split(" ")[0]
            col_names.append(name)
        else:
            col_name = body_name + " " + name
            col_names.append(col_name.strip())
    df.columns = col_names
            
    # change time from delta start time to actual timestamp
    for idx, row in df.iterrows():
        df.loc[idx, "Time"] = start_time + timedelta(seconds = float(df.loc[idx, "Time"]))
    timestamps = np.array([t.timestamp() * 1000 for t in df['Time']]).astype(int)

    obj_df = df.iloc[:, 3:]
    num_data_cols = 16 # 3 for position, 3 for rotation, 1 for residual, 9 for rotation matrix
    chunks = [obj_df.iloc[:, i:i+num_data_cols] for i in range(0, len(obj_df.columns), num_data_cols)]

    out = []
    for chunk in chunks:
        body = chunk.columns[0].split(" ")
        type, id = infer_type(body[0])
        chunk = np.array(chunk)
        for i, row in enumerate(chunk):
            position = row[0:3]
            if position[-1] == 0: # missing data, z == 0, skip
                continue
            # position = position / 1000 # convert to meters
            out.append({
                'time': int(timestamps[i]),
                'position': position.tolist(),
                'roll': row[3],
                'pitch': row[4],
                'yaw': row[5],
                'residual': row[6],
                'rotation': row[7:16].tolist(),
                'type': type,
                'id': id
            })
    return out
