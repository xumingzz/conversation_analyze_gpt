from typing import Dict, List
from .prompt import analyzer_prompt, summerize_prompt
from langchain.callbacks import StdOutCallbackHandler
from langchain.chat_models.base import BaseChatModel
from langchain.chains import LLMChain
from langchain.chains.mapreduce import MapReduceChain
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


class ConversationAnalyzer:
    """
    This class is used to analyze a conversation using a language model.
    """

    def __init__(
        self,
        chat: BaseChatModel,
        max_message_length: int = 1000,
        verbose: bool = False,
        need_summarize: bool = True,
    ):
        # Create the prompts for the language model.
        system_msg_prompt = SystemMessagePromptTemplate.from_template(
            analyzer_prompt["system"]
        )
        human_msg_prompt = HumanMessagePromptTemplate.from_template(
            analyzer_prompt["human"]
        )
        response_prompt = ChatPromptTemplate.from_messages(
            [system_msg_prompt, human_msg_prompt]
        )

        # Create the language model chain.
        self.normal_chain = LLMChain(llm=chat, prompt=response_prompt)
        self.max_message_length = max_message_length
        self.verbose = verbose
        self.need_summarize = need_summarize

        # Create the summarization chain.
        PROMPT = PromptTemplate(
            template=summerize_prompt["template"],
            input_variables=summerize_prompt["input_variables"],
        )
        self.summarize_chain = load_summarize_chain(
            llm=chat,
            chain_type="map_reduce",
            verbose=verbose,
            map_prompt=PROMPT,
            combine_prompt=PROMPT,
        )

    def parse_answer(self, data: str) -> Dict[str, List[str]]:
        """
        Parses the answer from the language model and returns a dictionary.

        Args:
        data (str): The response from the language model.

        Returns:
        Dict[str, List[str]]: A dictionary of responses, keyed by user ID.
        """
        result_dict = {}

        data_list = data.split("|||")

        for item in data_list:
            split_items = item.split(">>>")
            if len(split_items) < 3:
                continue
            user_id = split_items[1].strip()
            description = split_items[2].strip().replace("[", "").replace("]", "")
            if user_id in result_dict:
                result_dict[user_id].append(description)
            else:
                result_dict[user_id] = [description]

        return result_dict

    def analyze_conversation(self, text: str) -> Dict[str, List[str]]:
        """
        Analyzes a conversation using the language model.

        Args:
        text (str): The conversation to analyze.

        Returns:
        Dict[str, List[str]]: A dictionary of responses, keyed by user ID.
        """
        resp = {}
        if len(text) > self.max_message_length:
            # If the text is too long, split it into smaller chunks.
            lines = text.split("\n")
            subtext = ""
            for line in lines:
                if len(subtext) + len(line) <= self.max_message_length:
                    subtext += line + "\n"
                else:
                    sub_resp = self.analyze_conversation(subtext[:-1])
                    for key, value in sub_resp.items():
                        if key in resp:
                            resp[key].extend(value)
                        else:
                            resp[key] = value
                    subtext = line + "\n"
            if subtext:
                sub_resp = self.analyze_conversation(subtext[:-1])
                for key, value in sub_resp.items():
                    if key in resp:
                        resp[key].extend(value)
                    else:
                        resp[key] = value
            return resp
        else:
            # Run the language model chain on the text.
            response = self.normal_chain.run(
                msg=text, callbacks=[StdOutCallbackHandler()], verbose=self.verbose
            )
            resp = self.parse_answer(response)

        if self.need_summarize:
            for key, value in resp.items():
                resp[key] = [self.summarize_descriptions(value)]

        return resp

    def summarize_descriptions(self, descriptions: List[str]) -> str:
        """
        Summarizes a list of descriptions into a single summary.

        Args:
        descriptions (List[str]): A list of descriptions to summarize.

        Returns:
        str: The summarized description.
        """
        # Create a list of Document objects from the descriptions.
        docs = [Document(page_content=t) for t in descriptions[:3]]
        docs = []
        for i in range(0, len(descriptions), 3):
            combined_content = "\n".join(descriptions[i : i + 3])
            doc = Document(page_content=combined_content)
            docs.append(doc)

        # Run the summarization chain on the documents.
        return self.summarize_chain(
            {"input_documents": docs}, return_only_outputs=True
        )["output_text"]
