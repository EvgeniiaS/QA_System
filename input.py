import json
import retriever

def get_input():
    # Promts a user to enter the question, saves the retrieved document and the question to json file

    print("Enter your question below")
    question = input(">> ").lower()

    summary = retriever.get_document(question)
    
    text_question = {"passage": summary, "question": question}
    with open('text_question.json', 'w') as outfile:
        json.dump(text_question, outfile)

if __name__ == '__main__':
    get_input()