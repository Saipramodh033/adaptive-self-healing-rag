from src.config import load_settings
from src.dependencies import create_app_dependencies

def main():
    settings = load_settings()
    deps = create_app_dependencies(settings)
    vectorstore = deps.vectorstore
    
    queries = [
        "Something is wrong with my account and I cannot log in.",
        "Do you sell gaming consoles or video games?",
        "My laptop stopped turning on after 6 months. Should I return it or make a warranty claim?",
        "I bought a jacket but it does not fit. How do I return it and will it cost me anything?",
        "My payment went through but I never got an order confirmation. What should I do?",
        "My package says delivered but I never got it."
    ]
    
    for q in queries:
        print(f"\n{'='*50}\nQUERY: {q}\n{'='*50}")
        docs = vectorstore.similarity_search(q, k=3)
        for i, d in enumerate(docs):
            print(f"\n--- Doc {i+1} ---")
            print(d.page_content)

if __name__ == "__main__":
    main()
