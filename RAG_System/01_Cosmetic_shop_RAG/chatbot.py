

import faiss
import numpy as np
import pickle
import google.generativeai as genai

# --configuration--
API_KEY = "ENTER the api key here"
path_vector = "assets/Cosmetic_Product_Database.index"
path_pickle = "assets/Cosmetic_Product_backup pickle.pkl"
genai.configure(api_key=API_KEY)


class ShopBOT:
    def __init__(self):
        print("⏳ Loading System..")
        self.index = faiss.read_index(path_vector)
        with open(path_pickle, "rb") as f:
            self.df = pickle.load(f)

        # create an empty list to hold the conversation
        self.chat_history = []

        print("✅ System Ready!")

    def search_products(self, query,k=3):
        # 1. embed the query
        query_vector = genai.embed_content(
            model="gemini-embedding-001",
            content=query,
            task_type="retrieval_query"
        )["embedding"]

        # 2. search FAISS
        query_np = np.array([query_vector]).astype('float32')
        D,I = self.index.search(query_np, k)

        # 3. Get text
        result = []
        for row_idx in I[0]:
            if row_idx <len(self.df):
                item = self.df.loc[row_idx]['combined_text']
                result.append(item)
        return result

    def chat(self,user_query):
        # 1. Retrival
        # We search for the current question in the database
        relevant_products = self.search_products(user_query)
        context_text = "\n\n".join(relevant_products)

        # 2. format memory
        # we turn the list format in to a text string so the llm can read it
        memory_string = ""
        for speaker,text in self.chat_history:
            memory_string += f"{speaker}: {text}\n"
        model = genai.GenerativeModel("models/gemini-2.5-flash")

        prompt = f"""
                You are a smart cosmetic shop assistant. 
                Use the History and Context to answer the User.

                RULES:
                1. Look at the HISTORY to understand what we are talking about.
                2. Look at the CONTEXT to get product facts (Ingredients, Price, etc).
                3. If the user asks "How much is it?", check the History to see what "it" is.

                --- CONVERSATION HISTORY ---
                {memory_string}

                --- STORE INVENTORY (CONTEXT) ---
                {context_text}

                --- CURRENT INTERACTION ---
                User: {user_query}
                AI Answer:
                """

        response = model.generate_content(prompt)
        bot_reply = response.text

        # 4. update memory
        # add this readme.md turn to the history list
        self.chat_history.append(("AI", bot_reply))
        self.chat_history.append(("User", user_query))

        # Keep memory short (Optional: remove old messages if list gets too long)
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-20:]

        return bot_reply


if __name__ == "__main__":
    bot = ShopBOT()
    print("\n 💄 Chat with Memory! (Type 'exit' to quit)\n")

    while True:
        user_query = input("Customer: ")
        if user_query == "exit":
            break

        response = bot.chat(user_query)
        print(f"BOT: {response}")