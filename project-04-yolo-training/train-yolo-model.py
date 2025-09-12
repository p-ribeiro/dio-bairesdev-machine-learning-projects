

import torch
print(torch.cuda.is_available())      # True = GPU ready
print(torch.cuda.get_device_name(0))  # Shows GPU name


# %%
# downloading the dataset

from os import environ
from roboflow import Roboflow
from dotenv import load_dotenv
# from google.colab import userdata

load_dotenv()


rf = Roboflow(api_key=environ.get("roboflow_api_key"))
project = rf.workspace("hotwheels-znkle").project("hotwheels-tx3kw")
version = project.version(10)
dataset = version.download("yolov11")



# %% [markdown]
# Run the following code block to begin training. If you want to use a different model, number of epochs, or resolution, change `model`, `epochs`, or `imgsz`.

# %%
from ultralytics import YOLO

model = YOLO("yolo11s.pt")


model.train(data="HotWheels-10/data.yaml",
            epochs=100,
            imgsz=640,
            project="yolo",
            device=0
            )



# %%
model.predict(source="images/", project="results", imgsz=640, save=True)



