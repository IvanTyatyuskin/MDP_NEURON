import datetime

import uvicorn as uvicorn
from pydantic import BaseModel, ConfigDict
from fastapi import FastAPI
from src.api import downloadVideo
import os
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
    downloadVideo(user.event_id)
    #раскидать биометрию
    #запуск нейронки с передачей биометрии


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)
