from google import genai

# 1. Initialize the Gemini Client
# Set your key: export GOOGLE_API_KEY='your_key_here'
client = genai.Client(api_key="AIzaSyABwoDPRHAo0he5CazlsZ_WasGLrj7-p3c")

#Read the python file
with open("gemini_open_ai_compat.py", "r") as file:
    python_code = file.read()

prompt = f"Translate this Python Code to Java:\n{python_code}"

# 3. Call Gemini (using 1.5-Flash for speed or 1.5-Pro for complex logic)
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)

# 4. Print the result
print(response.text)