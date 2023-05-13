import os
from convasation_analyze_chain import ConvasationAnalyserChain
from convasation_analyze_chain import ConversationAnalyzer
from langchain.chat_models import ChatOpenAI, ChatAnthropic


def Demo():
    chat = ChatOpenAI()

    analyzer = ConversationAnalyzer(chat=chat)

    analyzser_chain = ConvasationAnalyserChain(analyzer=analyzer)

    analyzser_chain_resp = {}
    with open("demo.txt", "r", encoding="utf-8") as file:
        text = file.read()
        if len(text) > 0:
            analyzser_chain_resp = analyzser_chain(inputs={"text": text[:5000]})

    analyze_list = []
    if "text" in analyzser_chain_resp:
        analyze_list = analyzser_chain_resp["text"]

    print(analyze_list)


if "OPENAI_API_KEY" in os.environ:
    Demo()
