import requests
import os

# --- Configuration ---
# TODO: Replace these placeholder values with your actual details
ACCESS_TOKEN = "EAAR4pFKW5JEBPfPGhHcl7FrUiwZCguGZCKbk7W79sDjxKEWq8r8Gww3zPfDoRH7dzJLqFEQ8ZBJOqCKbo2ErKsl31stqja6X4XXaEaJcWl0VHUTpvUJKMaMUU9Xcj6dQCDSATZCmnszot3ZCvKKaBsmjeCSmeawDwXHEPJwhWMH1LN5jUYAwioZCFwq8J9Tj6gZAPGkeb8J7xlb7ftDLDxNoYqhgnE3yhfWUb8EiavZBFQZDZD"
PHONE_NUMBER_ID = "717383254790727"

# This is the exact name of your approved template
TEMPLATE_NAME = "start_conversation_prompt" 

# List of student phone numbers to send the message to
# Use the full number with country code, but without '+' or '00'
student_phone_numbers = [
    "916379613654",
    # "91xxxxxxxxxx", # Add more student numbers here
]

def send_start_template(to_number):
    """Sends the approved 'start' template message to a phone number."""
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "template",
        "template": {
            "name": TEMPLATE_NAME,
            "language": {
                "code": "en" # Your template is in English
            }
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        print(f"Successfully sent 'START' template to {to_number}.")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send template to {to_number}: {e}")
        print(f"Response: {response.text}")

# --- Main execution ---
if __name__ == "__main__":
    if ACCESS_TOKEN == "YOUR_FRESH_WHATSAPP_ACCESS_TOKEN_HERE" or PHONE_NUMBER_ID == "YOUR_PHONE_NUMBER_ID_HERE":
        print("!!! ERROR: Please replace the placeholder ACCESS_TOKEN and PHONE_NUMBER_ID before running.")
    else:
        for number in student_phone_numbers:
            send_start_template(number)
