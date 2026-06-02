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
import time
import datetime
import csv

import helper
import click

from transformers import LlavaProcessor, LlavaForConditionalGeneration

def parse_arguments():
    parser = argparse.ArgumentParser(description="Saliency implementation")
    parser.add_argument("data_path", help="")
    parser.add_argument("output_path", help="")
    parser.add_argument("config_path", help="")
    parser.add_argument(
        "--cpu", help="Use CPU instead of GPU.", action=argparse.BooleanOptionalAction
    )
    return parser.parse_args()

def get_files_in_folder(folder_path):
    try:
        files = []
        #files = [folder_path+"/"+f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        for f in os.listdir(folder_path):
            #print(folder_path+"/"+f)
            files.append(folder_path+"/"+f)
        return files
    except Exception as e:
        print(f"An error occurred while getting files in the folder: {str(e)}")
        return []

def avi_to_frames(video_path, output_folder):
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

	cap = cv2.VideoCapture(video_path)
	frame_count = 0

	while True:
		ret, frame = cap.read()
		if ret:
			frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
			cv2.imwrite(frame_filename, frame)
			frame_count += 1
		else:
			break

	cap.release()
	print(f"Extracted {frame_count} frames from {video_path} and saved them to {output_folder}")

def remove_symbols(string, spaces=False):
    pattern = r'[^a-zA-Z1-9]'
    if spaces: pattern = r'[^a-zA-Z1-9\ ]'
    cleaned_string = re.sub(pattern, '', string).lower()
    return cleaned_string

def encode_image(image_path, max_image=512):
	with Image.open(image_path) as img:
		width, height = img.size
		max_dim = max(width, height)
		if max_dim > max_image:
			scale_factor = max_image / max_dim
			new_width = int(width * scale_factor)
			new_height = int(height * scale_factor)
			img = img.resize((new_width, new_height))
		buffered = BytesIO()
		img.save(buffered, format="PNG")
		img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
		return img_str

def load_csv_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            data.append(row)
    return data
	
def write_csvline(content, path="results/", filename=None, date=False, different=False, extension="csv"):
    if date:
        filename = generate_numbers() + (" " + filename if filename else "")

    if not os.path.exists(path):
        os.makedirs(path) #recursively build a directory

    counter = 0
    tempfilename = path + filename + (" " + str(counter) if different else "") + "." + extension
    while os.path.isfile(tempfilename) and different:
        counter += 1
        tempfilename = path + filename + " " + str(counter) + "."+extension
    filename = tempfilename

    with open(filename, "a", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        writer.writerow(content)

    return filename[len(path):-4]

def main(
    data_path: str,
    output_path: str,
    config_path: str,
    method: str,
    use_cpu: bool = False):
    # Load config
    config = helper.load_yml(config_path)
    grid_config = config["grid_config"]
	
    # Set up CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    if use_cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise Exception(
            "No CUDA device found. This code requires a CUDA-enabled GPU and OpenCV with CUDA support."
        )
    
    # Load LLaVA processor and model
    model_directory = "llava_model\llava-1.5-7b-hf"
    processor = LlavaProcessor.from_pretrained(model_directory)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_directory,
        torch_dtype=torch.float16 if not use_cpu else torch.float32,
        low_cpu_mem_usage=True
    )
    
    device = "cuda" if torch.cuda.is_available() and not use_cpu else "cpu"
    model.to(device)

    frames_path = output_path+"/frames"
    os.makedirs(output_path, exist_ok=True)
    crops_path = frames_path + ("/crops")
    os.makedirs(crops_path, exist_ok=True)

    videos = get_files_in_folder(data_path)

    subset = helper.load_yml("config/smirk/crossing_26m.yml")
    
    for video in videos:
        #if not video.endswith(".avi"):
        #    continue
        video_id = video.split("/")[-1]
        video_path = video + "/" + video_id + ".avi"
        #os.makedirs(video, exist_ok=True)

        if video_id in subset:
             
            avi_to_frames(video_path, frames_path + "/" + video_id)
            images = get_files_in_folder(frames_path + "/" + video_id)
        
            for image in images:

                if not os.path.exists(crops_path+"/"+video_id):
                    os.makedirs(crops_path+"/"+video_id)

                frame_number = image.split("/")[-1][:-4]
            
                im = Image.open(image)
                cell_positions = helper.calculate_grid_cell_positions(im, grid_config)

                i=1
                for cell in cell_positions:
                    cropped_im = im.crop((cell[0][0], cell[0][1], cell[1][0], cell[1][1])) #select the right value in "cell" to match the desired format for crop()
                    cropped_im.save(crops_path+"/"+video_id+"/"+frame_number+"_region" + str(i) + ".png")
                    i=i+1
	
            #folder = video[:-4]
            #video = folder.split("/")[-1]
            write_csvline(["imagefile","prompt","response","boolresponse","elapsedtime"], filename=method+"_responses", path=output_path+"/"+video_id+"/")        

            #images = get_files_in_folder(folder)
            images.sort()
    
            crops = get_files_in_folder(crops_path + "/" + video_id)

            for image in crops:
			
                encoded_string = encode_image(image, 2048)
                prompt = "Is there a human pedestrian in this image? Answer ONLY either 'yes' or 'no'."

                if (method == "llava"):
                    #print("Using LLaVA API:", image)
                    #responseLlava = ollama.chat(model='llava:13b', messages=[{'role': 'user', 'content': prompt, 'images': [encoded_string]}])
                    #elapsedTime = responseLlava["total_duration"]
                    #responseGen = responseLlava['message']['content']
                    im = Image.open(image)
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image"},
                            ],
                        },
                    ]
                    prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)
                    inputs = processor(images=im, text=prompt_text, return_tensors="pt").to(device)

                    # Generate response from the model
                    start = time.time()
                    output_ids = model.generate(**inputs, max_new_tokens=100)
                    responseGen = processor.decode(output_ids[0], skip_special_tokens=True)
                    responseGen = responseGen[-4:]
                    end = time.time()
                    elapsedTime = int((end - start) * 1000000000)

                if (method == "gemma"):
                    #run the Gemma model through the Ollama API
                    start = time.time()
                    responseGemma = ollama.chat(model='gemma3:latest', messages=[{'role': 'user', 'content': prompt, 'images': [encoded_string]}])
                    elapsedTime = responseGemma["total_duration"]
                    responseGen = responseGemma['message']['content']
                    responseGen = responseGen[-4:]
                    end = time.time()
                    elapsedTime = int((end - start) * 1000000000)
                
                if (method == "minicpmv"):
                    #run the MiniCPM-v model through the Ollama API
                    start = time.time()
                    responseMinicpmv = ollama.chat(model='minicpm-v:latest', messages=[{'role': 'user', 'content': prompt, 'images': [encoded_string]}])
                    elapsedTime = responseMinicpmv["total_duration"]
                    responseGen = responseMinicpmv['message']['content']
                    responseGen = responseGen[-4:]
                    end = time.time()
                    elapsedTime = int((end - start) * 1000000000)
            
                if (method == "chatgpt"):
                    key = open("openai_key.txt", 'r').read()
                    client = OpenAI(api_key=key)

                    start = time.time()
                    response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}", }, }, ], }], max_tokens=300,)
                    responseGen = response.choices[0].message.content.replace("\n", " ")
                    end = time.time()
                    elapsedTime = int((end - start) * 1000000000)

                responseBool = 1 if "yes" in remove_symbols(responseGen) else 0	
                response = [image, prompt, responseGen, responseBool, elapsedTime]
                write_csvline(response, filename=method+"_responses", path=output_path+"/"+video_id+"/")

if __name__ == "__main__":
    args = parse_arguments()
    main(args.data_path, args.output_path, args.config_path, args.cpu)