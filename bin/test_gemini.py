from PIL import Image
import os
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

image = Image.open("../data/input/2024-08-07Lidl2_2.png")

model_gemini = genai.GenerativeModel(model_name="gemini-1.5-flash")
detect_prompt = '''Extract the text from the receipt.'''
prompt_extract= '''Given the receipt text, identify the date, 'total_amount', merchant, and items ("name", "number", "price"). If you can't identify any of these, leave it blank.'''
prompt_json = '''
Given the text extracted create a JSON object with the following structure:
{
    "date": "",
    "total_amount": "",
    "merchant": "",
    "items": [
        {
            "name": "",
            "number": "",
            "price": ""
        }
    ]
}

Rules:
- If any field is missing or appears incorrect, leave it as an empty string
- Do not include any comments in the JSON
- Ensure the JSON is valid and well-formatted
- Respond ONLY with the JSON object
'''

def analyze_with_gemini(image: Image, prompt) -> str:
        try:
            response = model_gemini.generate_content([image, prompt])
            return str(response.text)
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return None
        

text = analyze_with_gemini(image, detect_prompt)
print(text)
extracted = analyze_with_gemini(text, prompt_extract)
print(extracted)
json = analyze_with_gemini(extracted, prompt_json)
print(json)