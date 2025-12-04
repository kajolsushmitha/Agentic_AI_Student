import os
import pytesseract
from PIL import Image
import datetime

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# --- Configuration ---
# Set the path to the Tesseract executable (change this if it's not in your PATH)
# Example for Windows: r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Example for macOS/Linux (often not needed if installed via brew/apt and in PATH): '/usr/local/bin/tesseract'
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\jahag\AppData\Local\Programs\Tesseract-OCR\tesseract.exe" # <--- IMPORTANT: Update this path!

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/calendar.events']

def authenticate_google_calendar():
    """Shows user how to authenticate and returns Google Calendar service."""
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    try:
        service = build('calendar', 'v3', credentials=creds)
        return service
    except HttpError as error:
        print(f'An error occurred: {error}')
        return None

def extract_text_from_image(image_path):
    """
    Extracts text from an image file using Tesseract OCR.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text
    except pytesseract.TesseractNotFoundError:
        print("Error: Tesseract is not installed or not in your PATH. Please install it.")
        print("For Windows, ensure 'pytesseract.pytesseract.tesseract_cmd' is set correctly.")
        return None
    except Exception as e:
        print(f"An error occurred during OCR: {e}")
        return None

def create_google_calendar_event(service, extracted_text):
    """
    Creates a new event in Google Calendar using the extracted text.
    You will likely need to parse the extracted_text to get meaningful event details.
    For simplicity, this example puts the entire extracted_text into the description.
    """
    if not service:
        print("Google Calendar service not available.")
        return

    # --- IMPORTANT: How to parse your extracted_text ---
    # This is the most challenging part. OCR is not perfect, and parsing unstructured
    # text into structured event details (summary, start/end time, location)
    # requires robust logic.

    # For this example, we'll use a placeholder summary and put the full text in description.
    # You would need to implement logic here to:
    # 1. Identify event title/summary
    # 2. Identify date and time (e.g., "July 25th at 3 PM") and convert to RFC3339 format
    # 3. Identify location
    # 4. Identify description details

    # Example of a basic event structure
    event_summary = "Event from Image (Review Details!)"
    event_description = f"Text extracted from image:\n\n{extracted_text}"

    # For simplicity, we'll set a generic future event for demonstration.
    # You MUST parse the date and time from your `extracted_text` for real-world use.
    # Example: Event 2 hours from now, lasting 1 hour.
    now = datetime.datetime.utcnow()
    start_time = now + datetime.timedelta(hours=2)
    end_time = start_time + datetime.timedelta(hours=1)

    event = {
        'summary': event_summary,
        'location': 'Online/To Be Determined', # You'd parse this from extracted_text
        'description': event_description,
        'start': {
            'dateTime': start_time.isoformat() + 'Z', # 'Z' indicates UTC time
            'timeZone': 'Asia/Kolkata', # Set your desired timezone
        },
        'end': {
            'dateTime': end_time.isoformat() + 'Z',
            'timeZone': 'Asia/Kolkata', # Set your desired timezone
        },
        'reminders': {
            'useDefault': False,
            'overrides': [
                {'method': 'email', 'minutes': 24 * 60},
                {'method': 'popup', 'minutes': 10},
            ],
        },
    }

    try:
        # 'primary' refers to the default calendar. You can also specify a calendar ID.
        event = service.events().insert(calendarId='primary', body=event).execute()
        print(f'Event created: {event.get("htmlLink")}')
    except HttpError as error:
        print(f'An error occurred creating the event: {error}')

def main():
    image_file_path = r"C:\Users\jahag\Desktop\MAIN_PROJECT\Agentic_Ai\OS_subject\cal.png" # <--- IMPORTANT: Change this to your image file name!

    print(f"Attempting to extract text from: {image_file_path}")
    extracted_text = extract_text_from_image(image_file_path)

    if extracted_text:
        print("\n--- Extracted Text ---")
        print(extracted_text)
        print("----------------------")

        # 2. Authenticate with Google Calendar
        print("\nAuthenticating with Google Calendar...")
        service = authenticate_google_calendar()

        # 3. Create Google Calendar event
        if service:
            print("\nCreating Google Calendar event...")
            create_google_calendar_event(service, extracted_text)
        else:
            print("Failed to authenticate Google Calendar. Event not created.")
    else:
        print("No text extracted. Cannot create event.")

if __name__ == '__main__':
    main()