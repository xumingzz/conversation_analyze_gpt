# Conversation Analyzer Chain

The Conversation Analyzer Chain is a Python package that provides a Chain for analyzing conversation. It uses a ConversationAnalyzer object to analyze the input text and returns the results.

## API

### `ConvasationAnalyserChain`

A Chain for analyzing conversation.

#### `_call`

Runs the language model on the input text and returns the result.

```python
def _call(
    self,
    inputs: Dict[str, str],
    run_manager: Optional[CallbackManagerForChainRun] = None,
) -> Dict[str, dict]:
```

#### `_acall`

Runs the language model asynchronously on the input text and returns the result.

```python
async def _acall(
    self,
    inputs: Dict[str, str],
    run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
) -> Dict[str, dict]:
```
