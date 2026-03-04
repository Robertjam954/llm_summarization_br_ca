from document_processor import LuciDocumentProcessor
from vector_store_manager import LuciVectorStoreManager
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatCohere
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
from config import Config


class LuciSummarizer:
    """
    Handles document summarization using various AI models.
    """

    def __init__(self, document_file, summary_file):
        """
        Initializes the LuciSummarizer with the document and summary files.

        :param document_file: The uploaded file of the document to be summarized.
        :param summary_file: The uploaded file of the golden summary.
        """
        self.document_file = document_file
        self.summary_file = summary_file
        self.document_text = ""
        self.summary_text_golden = ""
        self.summarised_text = ""

    def process_documents(self):
        """
        Extracts text from the document and summary files and initializes vector stores.
        """
        # Process and extract text from the document
        self.document_text = (
            LuciDocumentProcessor(self.document_file).get_text()
            if self.document_file
            else ""
        )
        # Process and extract text from the summary
        self.summary_text = (
            LuciDocumentProcessor(self.summary_file).get_text()
            if self.summary_file
            else ""
        )

        # Initialize vector stores for the document and summary
        self.document_vector_store = LuciVectorStoreManager(
            self.document_text
        ).build_vectorstore()
        self.summary_vector_store = LuciVectorStoreManager(
            self.summary_text
        ).build_vectorstore()

        return self.document_text, self.summary_text

    def summarise_documents(self, model="gpt-4", task="Summarize"):
        # Modify the prompt based on the task
        if task == "Summarize":
            prompt_template = (
                "Please summarize this document in exactly four paragraphs: {article}"
            )
        else:
            prompt_template = (
                "Please follow the instructions in this document {article}"
            )

        prompt = ChatPromptTemplate.from_template(prompt_template)
        output_parser = StrOutputParser()
        model = self._get_model_interface(model)
        chain = {"article": RunnablePassthrough()} | prompt | model | output_parser

        summarised_text = chain.invoke(self.document_text)

        return summarised_text

    def _get_model_interface(self, model_name):
        if model_name == "GPT-4":
            return ChatOpenAI(model="gpt-4", openai_api_key=Config.OPENAI_API_KEY)
        if model_name == "GPT-3":
            return ChatOpenAI(openai_api_key=Config.OPENAI_API_KEY)
        elif model_name == "Claude-2":
            # Example using another API key
            return ChatAnthropic(
                model="claude-2", anthropic_api_key=Config.ANTHROPIC_API_KEY
            )
        elif model_name == "Claude-3":
            # Example using another API key
            return ChatAnthropic(
                model="claude-3-sonnet-20240229",
                anthropic_api_key=Config.ANTHROPIC_API_KEY,
            )
        elif model_name == "Cohere":
            # Example using another API key
            return ChatCohere(cohere_api_key=Config.COHERE_API_KEY)
        elif model_name == "Luci-FT-AM":
            # Example using another API key
            return ChatOpenAI(
                model="ft:gpt-3.5-turbo-0613:personal::8GQ4E4S4",
                openai_api_key=Config.OPENAI_API_KEY,
            )
        else:
            return Ollama(model="llama2")
