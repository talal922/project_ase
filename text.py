


from google import genai

# Initialize client
client = genai.Client(api_key="AIzaSyABZpAFuvry5bUkEwzOMuyUoP5rYsdpIvM")

# Generate text
response = client.models.generate_content(
    model="gemini-3-flash-preview",  # choose a valid model
    contents="Explain how AI works in simple terms."
)

# Output
print(response.text)
