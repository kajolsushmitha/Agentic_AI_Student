from __future__ import print_function
import os
import google.auth.transport.requests
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# SCOPES for Classroom and Drive
SCOPES = [
    'https://www.googleapis.com/auth/classroom.courses',
    'https://www.googleapis.com/auth/classroom.coursework.students',
    'https://www.googleapis.com/auth/drive.file'
]

# Authenticate user
def authenticate():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(google.auth.transport.requests.Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

# Upload file to Google Drive
def upload_file_to_drive(service_drive, file_path):
    file_metadata = {'name': os.path.basename(file_path)}
    media = MediaFileUpload(file_path, mimetype='application/pdf')
    file = service_drive.files().create(body=file_metadata, media_body=media, fields='id').execute()
    return file.get('id')

# Post assignment in Classroom
def post_assignment(service_classroom, course_id, title, drive_file_id):
    coursework = {
        'title': title,
        'materials': [
            {
                'driveFile': {
                    'driveFile': {
                        'id': drive_file_id
                    },
                    'shareMode': 'VIEW'
                }
            }
        ],
        'workType': 'ASSIGNMENT',
        'state': 'PUBLISHED'
    }
    coursework = service_classroom.courses().courseWork().create(
        courseId=course_id,
        body=coursework
    ).execute()
    print(f"✅ Assignment created: {coursework.get('title')}")

if __name__ == '__main__':
    creds = authenticate()

    # Build services
    service_drive = build('drive', 'v3', credentials=creds)
    service_classroom = build('classroom', 'v1', credentials=creds)

    # Hardcode COURSE_ID for Agent OS
    COURSE_ID = '791255014049'  # Agent OS course
    FILE_PATH = 'sy.pdf'        # PDF to upload

    # Upload to Drive
    file_id = upload_file_to_drive(service_drive, FILE_PATH)

    # Create assignment in Classroom
    post_assignment(service_classroom, COURSE_ID, "Uploaded PDF Assignment", file_id)
