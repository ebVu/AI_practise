import glob
import sys
import os
import csv

database_path = "/media/sf_RZV/DeepLearning/group_practise/database/UTKFace"

fields = ['Filename', 'Age', 'Gender', 'Race']
database_csv = 'UTKFace.csv'
with open(database_csv, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields) 
    for f in glob.glob(os.path.join(database_path, "*.jpg")):
        filename = os.path.basename(f)
        file_parts = filename.split('_')
        row = [f, file_parts[0], file_parts[1], file_parts[2]]
        csvwriter.writerow(row)
