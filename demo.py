from convasation_analyze_chain import ConvasationAnalyserChain
from analyzer.conversation_analyze import ConversationAnalyzer
from langchain.llms.fake import FakeListLLM


def Demo():
    responses = [
        """
        >>>[用户1]>>>[用户1是一个好人]|||
        >>>[用户2]>>>[用户2是一个坏人]|||
        """,
        "总结=用户1是一个好人",
        "总结=用户1是一个大好人",
        "总结=用户2是一个好人",
        "总结=用户2是一个大好人",
        "summarizes",
    ]
    chat = FakeListLLM(responses=responses)

    analyzer = ConversationAnalyzer(chat=chat, max_message_length=1000)

    analyzser_chain = ConvasationAnalyserChain(
        conversation_analyzer=analyzer, verbose=True
    )

    analyzser_chain_resp = {}
    with open("demo.txt", "r", encoding="utf-8") as file:
        text = file.read()
        if len(text) > 0:
            analyzser_chain_resp = analyzser_chain(inputs={"text": text[:5000]})

    analyze_list = []
    if "text" in analyzser_chain_resp:
        analyze_list = analyzser_chain_resp["text"]

    print(analyze_list)


Demo()
