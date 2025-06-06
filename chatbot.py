import json
import random
import pickle
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENTS_PATH = os.path.join(BASE_DIR, 'data', 'intents.json')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'intent_classifier.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'models', 'vectorizer.pkl')

# Load the intents JSON
with open(INTENTS_PATH, 'r') as file:
    intents = json.load(file)

# Load trained model and vectorizer
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, 'rb') as f:
    vectorizer = pickle.load(f)

def get_response(user_input):
    """
    Predict intent from user_input and return a random response.
    """
  
    user_input_vect = vectorizer.transform([user_input])
    
   
    intent_tag = model.predict(user_input_vect)[0]

   
    for intent in intents['intents']:
        if intent['tag'] == intent_tag:
            return random.choice(intent['responses'])
    

    return "Sorry, I didn't understand that."

if __name__ == "__main__":
    print("College Chatbot is running! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("Chatbot: Goodbye!")
            break
        response = get_response(user_input)
        print(f"Chatbot: {response}")
