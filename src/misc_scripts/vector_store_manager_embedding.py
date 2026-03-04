import os

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, OpenAI


class LuciVectorStoreManager:
    """
    Manages the creation and retrieval of vector stores for text data.
    """

    def __init__(self, text):
        """
        Initializes the LuciVectorStoreManager with text to be vectorized.

        :param text: The text to build the vector store from.
        """
        self.text = text
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

    def build_vectorstore(self):
        """
        Builds and returns a vector store from the provided text.

        :return: A retriever object for the constructed vector store.
        """
        vectorstore = FAISS.from_texts(
            [self.text], embedding=OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        )
        return vectorstore.as_retriever()
