import datetime

import uvicorn as uvicorn
from pydantic import BaseModel, ConfigDict
from fastapi import FastAPI
from api import downloadVideo
from neuron import startNeuralNetwork
import os
import glob
import shutil

class Biometrics(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    photo_id: str
    upload_date: datetime.date
    photo_path: str
    employee_id: str

class StartAIModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    event_id: str
    biometrics: list[Biometrics]

app = FastAPI()

def createFacesDataset(facesArray: list[Biometrics]):
    os.chdir('src/faces-dataset')
    os.rmdir()
    alreadyCreated = []

    for elem in facesArray:
        if elem.employee_id not in alreadyCreated:
            os.mkdir(f"/{elem.employee_id}")

        shutil.copyfile(elem.photo_path, f"src/{elem.employee_id}/{elem.photo_id}")


@app.post("/startAI")
def start(user: StartAIModel):
    createFacesDataset(user.biometrics)
    downloadVideo(user.event_id)

    list_of_files = glob.glob('src/videos/*.mp4')
    latest_file = max(list_of_files, key=os.path.getctime)

    startNeuralNetwork(latest_file, user.event_id)

if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)
