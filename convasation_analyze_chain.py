from typing import Dict, List, Optional

from pydantic import Extra

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain

from analyzer.conversation_analyze import ConversationAnalyzer


class ConvasationAnalyserChain(Chain):
    output_keys: List[str] = ["text"]
    conversation_analyzer: ConversationAnalyzer

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @property
    def input_keys(self) -> List[str]:
        """Returns the input keys expected by the prompt."""
        return ["text"]

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, dict]:
        """Runs the language model on the input text and returns the result."""
        if "text" not in inputs:
            raise ValueError("The input parameters must contain a 'text' key")

        result = self.conversation_analyzer.analyze_conversation(inputs["text"])

        if run_manager:
            run_manager.on_text("Log something about this run")

        return {self.output_keys[0]: result}

    async def _acall(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, dict]:
        """Runs the language model asynchronously on the input text and returns the result."""
        if "text" not in inputs:
            raise ValueError("The input parameters must contain a 'text' key")

        result = await self.conversation_analyzer.analyze_conversation(inputs["text"])

        if run_manager:
            await run_manager.on_text("Log something about this run")

        return {self.output_keys[0]: result}

    @property
    def _chain_type(self) -> str:
        return "convasation_analyzer_chain"
