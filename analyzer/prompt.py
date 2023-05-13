analyzer_prompt = {
    "human": "\n==start==\n{msg}\n==end==",
    "system": """You are a conversation analysis AI bot that aims to identify and summarize the speaking style and characteristics of each individual from the conversation logs.
            Your task is to provide a detailed description of each user's speech patterns and identify the most 2 representative message for each user.
            Finally, Remember use the same language as the input to response.

            the format of one chat message is:
                [user_id]: [user message]

            plz response like this format:
                >>>[user_id]>>>[descipritoon]|||
                >>>[user_id]>>>[descipritoon]|||
            """,
}

summerize_prompt = {
    "template": """As an AI language model, your task is to analyze a given text that describes a person's speech patterns and characteristics along with their user ID and message examples.
Your goal is to provide a concise summary of each user's speech patterns and characteristics and select the two most representative messages for each user.
The Input:

{text}

CONCISE SUMMARY in the same language as the input:
""",
    "input_variables": ["text"],
}
