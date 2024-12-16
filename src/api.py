import requests

BASE_URL = "http://localhost:8000"

class Token():
    access_token: str
    token_type: str

class Visit():
    employee_id: str | None = None
    event_id: str

ACCESS_TOKEN = Token()

def login():
    result = requests.post(BASE_URL + "/login", {'login': 'string', 'password_hash:': 'string'}, headers={"Authorization" : f"Bearer {ACCESS_TOKEN.access_token}"}).json()

    ACCESS_TOKEN.access_token = result['access_token']
    ACCESS_TOKEN.token_type = result['token_type']

def visitCreate(employee_id:str, event_id: str):
    result = requests.post(BASE_URL + "/visit", {'employee_id': employee_id, 'event_id': event_id}, headers={"Authorization" : f"Bearer {ACCESS_TOKEN.access_token}"}).json()
    if (result.status_code == 401):
        login()
        visitCreate(employee_id, event_id)

    return result['visit_id']

def visitMark(visit_id: str, filePath: str):
    with open(filePath, 'rb') as file:
        result = requests.post(BASE_URL + "/visit-marks", {'visit_id': visit_id, 'file': file}, headers={"Authorization" : f"Bearer {ACCESS_TOKEN.access_token}"})

    if (result.status_code == 401):
        login()
        visitMark(visit_id, filePath)

def downloadVideo(event_id: str):
    result = requests.get(BASE_URL + "/download-event-video", {'event_id': event_id}, headers={"Authorization" : f"Bearer {ACCESS_TOKEN.access_token}"})

    if (result.status_code == 401):
        login()
        downloadVideo(event_id)

    with open('src/videos', "wb") as path:
        path.write(result)

