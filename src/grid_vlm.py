import ollama
from openai import OpenAI
import os
import re
from io import BytesIO
import base64
import cv2
from PIL import Image
import argparse
import torch
import numpy as np

import datetime
import csv

import helper

def parse_arguments():
    parser = argparse.ArgumentParser(description="VLM implementation")
    parser.add_argument("data_path", help="")
    parser.add_argument("output_path", help="")
    parser.add_argument("config_path", help="")
    parser.add_argument("use_flickering_handling", help="")
    parser.add_argument(
        "--cpu", help="Use CPU instead of GPU.", action=argparse.BooleanOptionalAction
    )
    return parser.parse_args()

def get_files_in_folder(folder_path):
    try:
        files = []
        for f in os.listdir(folder_path):
            files.append(folder_path+"/"+f)
        return files
    except Exception as e:
        print(f"An error occurred while getting files in the folder: {str(e)}")
        return []

def load_csv_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.DictReader(file)#, delimiter=";")
        for row in csv_reader:
            data.append(row)
    return data

def read_csvcontents(filename,delimiter=","):
    retu = []
    with open(filename, 'r') as data:
        for line in csv.reader(data, delimiter=delimiter):
            retu += [line]
    return retu

def main(
    data_path: str,
    output_path: str,
    config_path: str,
    use_flickering_handling: bool,
    use_cpu: bool = False):
    # Load config
    config = helper.load_yml(config_path)
    grid_config = config["grid_config"]

    frames_path = output_path+"/frames"
    os.makedirs(output_path, exist_ok=True)
    crops_path = frames_path + ("/crops")
    os.makedirs(crops_path, exist_ok=True)

    videos = get_files_in_folder(data_path)

    subset = helper.load_yml("config/smirk/crossing_26m.yml")

    for video in videos:

        video_id = video.split("/")[-1]
        video_path = video + "/" + video_id + ".avi"

        if video_id in subset:

            vlm_responses = []
            files = get_files_in_folder(output_path+"/"+video_id)
            for file in files:
                if "vlm_responses.csv" in file:
                    vlm_responses = load_csv_file(file)
                    #vlm_responses = read_csvcontents(vlm_responses, delimiter=",")
                    break
                elif "chatgpt_responses.csv" in file:
                    vlm_responses = load_csv_file(file)
                    break

            maxframe = 0
            for vlm in vlm_responses:
                image_file = vlm["imagefile;prompt;response;boolresponse;elapsedtime"].split(";")[0]
                lp = image_file.split("/")[-1][:-4].split("_")
                vlm["frame"] = int(lp[1])
                vlm["cell"] = int(lp[2][6:])
                if vlm["frame"] > maxframe:
                    maxframe = vlm["frame"]
            cell_values = {}
            for f in range(1, maxframe+1):
                row = []
                for c in [1,2,3,4,5,6]:
                    found = False
                    for response in vlm_responses:
                        if response["frame"] == f and response["cell"] == c:
                            #boolean_response = response["boolresponse"]
                            boolean_response = response["imagefile;prompt;response;boolresponse;elapsedtime"].split(";")[3]
                            row.append(int(boolean_response))
                            found = True
                            break
                    if not found:
                        row.append(-1)
                cell_values[response["frame"]] = row

            if (use_flickering_handling):# Flickering effect handling
                max_gap = 4
                #while (frame <= len(cell_values)-1):
                for frame in range (1, len(cell_values)-1):
                    for cell in range (len(cell_values[frame])):
                        if (cell_values[frame][cell]==1):
                            if (0<cell<len(cell_values[frame])-1): # continue with the middle cells
                                if (cell_values[frame+1][cell-1]==cell_values[frame+1][cell]==cell_values[frame+1][cell+1]==0): # check if the next cell is 0 and if there is no pedestrian detected on the adjacent cells
                                    coming_cells = 0
                                    for i in range(frame+2, frame+max_gap+1):
                                        coming_cells = coming_cells + cell_values[i][cell-1]+cell_values[i][cell]+cell_values[i][cell+1]
                                    if (coming_cells >= 1):
                                        cell_values[frame+1][cell]=1
                            elif (cell==0): # continue with the first cell
                                if (cell_values[frame+1][cell]==cell_values[frame+1][cell+1]==0): # check if the next cell is 0 and if there is no pedestrian detected on the cell second cell
                                    coming_cells = 0
                                    for i in range(frame+2, frame+max_gap+1):
                                        coming_cells = coming_cells + cell_values[i][cell]+cell_values[i][cell+1]
                                    if (coming_cells >= 1):
                                        cell_values[frame+1][cell]=1
                            else: # continue with the last cell
                                if (cell_values[frame+1][cell-1]==cell_values[frame+1][cell]==0): # check if the next cell is 0 and if there is no pedestrian detected on the previous cell
                                    coming_cells = 0
                                    for i in range(frame+2, frame+max_gap+1):
                                        coming_cells = coming_cells + cell_values[i][cell-1]+cell_values[i][cell]
                                    if (coming_cells >= 1):
                                        cell_values[frame+1][cell]=1
                    #frame = frame + 1
                    if (frame+max_gap+1>len(cell_values)):
                        max_gap = max_gap-1
            
            helper.save_cell_value_csv(cell_values, output_path+"/"+video_id, grid_config)