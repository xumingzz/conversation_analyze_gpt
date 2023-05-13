from typing import Dict, List
from langchain.chains import LLMChain
from langchain.callbacks import StdOutCallbackHandler
from langchain.chat_models.base import BaseChatModel
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from prompt import analyzer_prompt


class ConversationAnalyzer:
    def __init__(
        self,
        chat: BaseChatModel,
        prompt_templates: Dict[str, str] = analyzer_prompt,
        max_message_length: int = 1000,
    ):
        system_msg_prompt = SystemMessagePromptTemplate.from_template(
            prompt_templates["system"]
        )

        human_msg_prompt = HumanMessagePromptTemplate.from_template(
            prompt_templates["human"]
        )

        response_prompt = ChatPromptTemplate.from_messages(
            [system_msg_prompt, human_msg_prompt]
        )

        self.chain = LLMChain(llm=chat, prompt=response_prompt)
        self.max_message_length = max_message_length

    def parse_answer(self, data: str) -> Dict[str, List[str]]:
        result_dict = {}

        data_list = data.split("|||")

        for item in data_list:
            user_id = item.split(">>>")[1].strip()
            description = item.split(">>>")[2].strip().replace("[", "").replace("]", "")
            if user_id in result_dict:
                result_dict[user_id].append(description)
            else:
                result_dict[user_id] = [description]

        return result_dict

    def analyze_conversation(self, text: str) -> Dict[str, List[str]]:
        if len(text) > self.max_message_length:
            resp = {}
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
            response = self.chain.run(
                msg=text, callbacks=[StdOutCallbackHandler()], verbose=True
            )
            return self.parse_answer(response)
