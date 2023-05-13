import os
from convasation_analyze_chain import ConvasationAnalyserChain
from convasation_analyze_chain import ConversationAnalyzer
from langchain.chat_models import ChatOpenAI, ChatAnthropic


def Demo():
    chat = ChatOpenAI()

    analyzer = ConversationAnalyzer(chat=chat)

    msg_analyzser_chain = ConvasationAnalyserChain(analyzer=analyzer)

    msg_analyzser_chain_resp = {}
    with open("demo.txt", "r", encoding="utf-8") as file:
        # 文件操作代码
        text = file.read()
        if len(text) > 0:
            msg_analyzser_chain_resp = msg_analyzser_chain(inputs={"text": text[:5000]})

    analyze_list = []
    if "text" in msg_analyzser_chain_resp:
        analyze_list = msg_analyzser_chain_resp["text"]

    print(analyze_list)


if "OPENAI_API_KEY" in os.environ:
    Demo()
