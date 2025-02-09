# import os
# import streamlit as st
# from langchain_groq import ChatGroq
# from langchain_core.prompts import PromptTemplate
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Initialize LLaMA connection via Groq
# class LlamaChain:
#     def __init__(self):
#         self.llm = ChatGroq(
#             temperature=0,
#             groq_api_key=os.getenv("GROQ_API_KEY"),
#             model_name="llama-3.3-70b-versatile"
#         )

#     def get_response(self, user_input):
#         # Create a prompt for the LLaMA model
#         prompt = PromptTemplate.from_template(
#             """
#             ### USER QUERY:
#             {query}
#             ### INSTRUCTION:
#             Respond to the user's query in a conversational and helpful manner.
#             """
#         )
#         # Generate response
#         chain = prompt | self.llm
#         response = chain.invoke({"query": user_input})
#         return response.content

# # Streamlit UI
# st.title("🤖 Chatbot Powered by LLaMA via Groq")
# st.subheader("Type your question below and get an intelligent response!")

# # Initialize LLaMA chain
# llama_chain = LlamaChain()

# # User input
# user_query = st.text_input("Enter your query:")

# # Submit button
# if st.button("Submit"):
#     if user_query.strip():
#         # Display a spinner while processing
#         with st.spinner("Fetching response..."):
#             try:
#                 response = llama_chain.get_response(user_query)
#                 st.success("Response from LLaMA:")
#                 st.write(response)
#             except Exception as e:
#                 st.error(f"An error occurred: {e}")
#     else:
#         st.warning("Please enter a valid query.")
