# %%
import os
os.environ["GOOGLE_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""

# %%
# !pip install aiohttp -q

# %%
import os
import aiohttp
import asyncio
import json
from typing import Any, List, Optional, TypedDict
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import LLMResult
from langchain_core.callbacks.manager import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from pydantic import Field
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
# LangGraph Imports
from langgraph.graph import StateGraph, END

# %%
# --- 1. Async Custom LLM Implementation ---

class AsyncCustomHTTPGemini(BaseLLM):
    """
    An async custom LangChain LLM wrapper that interacts with the Google Gemini API
    using aiohttp for asynchronous HTTP requests.
    """

    # Model and API Configuration
    api_key: Optional[str] = None
    model_name: str = Field(default="gemini-2.5-flash", alias="model")
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/models/"
    response_schema: Optional[dict] = None
    _session: Optional[aiohttp.ClientSession] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if not self.api_key:
            self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY must be provided or set as an environment variable.")

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "async_custom_http_gemini"

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Async HTTP POST request to the Gemini API.
        """
        # 1. Construct the API Endpoint
        api_endpoint = f"{self.base_url}{self.model_name}:generateContent"
        url = f"{api_endpoint}?key={self.api_key}"
        headers = {"Content-Type": "application/json"}

        # 2. Construct the request body
        request_data = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }

        # 3. Add generation config for JSON output if schema is present
        if self.response_schema:
            request_data["generationConfig"] = {
                "responseMimeType": "application/json",
                "responseSchema": self.response_schema
            }

        # 4. Send async request
        try:
            session = await self._get_session()
            async with session.post(url, headers=headers, json=request_data) as response:
                response.raise_for_status()
                response_json = await response.json()
                generated_text = response_json['candidates'][0]['content']['parts'][0]['text']
                return generated_text

        except aiohttp.ClientResponseError as err:
            error_message = f"Gemini API HTTP Error ({err.status}): {err.message}"
            raise RuntimeError(error_message) from err
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred during async API call: {e}")

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Sync fallback - runs the async version in an event loop."""
        return asyncio.get_event_loop().run_until_complete(
            self._acall(prompt, stop, None, **kwargs)
        )

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Async generation for multiple prompts."""
        generations = []
        for prompt in prompts:
            text = await self._acall(prompt, stop, run_manager, **kwargs)
            generations.append([{"text": text}])
        return LLMResult(generations=generations)

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Sync generate fallback."""
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop, run_manager, **kwargs)
            generations.append([{"text": text}])
        return LLMResult(generations=generations)

    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

# %%
# --- 2. LangGraph AGENT DEFINITIONS (Async) ---

# 2.1. Define the State for the Graph
class AgentState(TypedDict):
    """
    The state of the graph, holding data passed between nodes.
    """
    item_name: str
    complexity_level: str
    score: Optional[float]
    review_data: Optional[str]


# --- JSON Schema Definitions (Mandatory for Gemini JSON mode) ---

# Schema for Agent 1: Score
SCORE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "score": {"type": "NUMBER", "description": "A random technical score between 0.0 and 1.0."},
    },
    "required": ["score"],
    "propertyOrdering": ["score"]
}

# Schema for Agent 2: Detailed Review
REVIEW_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "review_text": {"type": "STRING", "description": "A concise, technical review."},
        "category": {"type": "STRING", "description": "The category of the review (e.g., 'Positive', 'Neutral', 'Negative')."}
    },
    "required": ["review_text", "category"],
    "propertyOrdering": ["review_text", "category"]
}


# 2.2. Async Node for Agent 1: Score Generator
async def score_generator_node(state: AgentState) -> dict:
    """
    Async: Generates a technical score (0.0 to 1.0) for the item.
    """
    print("--- [Agent 1] Executing: Score Generator (ASYNC) ---")

    # 1. Initialize async LLM with Score Schema
    llm_score_generator = AsyncCustomHTTPGemini(model_name="gemini-2.5-flash", response_schema=SCORE_SCHEMA)

    # 2. Construct Prompt
    prompt_1 = PromptTemplate.from_template(
        "You are a technical analyst. Your task is to assign a random score between 0.0 and 1.0 to the '{item_name}' based on its complexity '{complexity_level}'. Output the result strictly in JSON format according to the schema."
    )
    prompt_value = prompt_1.format(
        item_name=state['item_name'],
        complexity_level=state['complexity_level']
    )

    # 3. Call LLM asynchronously using _acall
    raw_json_output = await llm_score_generator._acall(prompt_value)
    await llm_score_generator.close()

    # 4. Parse JSON and update state
    try:
        score_data = json.loads(raw_json_output)
        score = score_data.get('score', 0.0)
        print(f"--- [Agent 1] Generated Score: {score} ---")
        return {"score": score}
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from Agent 1: {e}. Falling back to score 0.5.")
        return {"score": 0.5}


# 2.3. Async Node for Agent 2: Review Generator
async def review_generator_node(state: AgentState) -> dict:
    """
    Async: Generates a detailed review based on the generated score and context.
    """
    print("\n--- [Agent 2] Executing: Review Generator (ASYNC) ---")

    # 1. Initialize async LLM with Review Schema
    llm_review_generator = AsyncCustomHTTPGemini(model_name="gemini-2.5-flash", response_schema=REVIEW_SCHEMA)

    # 2. Construct Prompt
    prompt_2 = PromptTemplate.from_template(
        "Generate a technical review for the item '{item_name}' which has a complexity of '{complexity_level}' and received a technical score of {score}. Your review must reflect this score. Output the review strictly in JSON format according to the schema."
    )

    prompt_value = prompt_2.format(
        item_name=state['item_name'],
        complexity_level=state['complexity_level'],
        score=state['score']
    )

    # 3. Call LLM asynchronously using _acall
    raw_json_output = await llm_review_generator._acall(prompt_value)
    await llm_review_generator.close()

    # 4. Update state with the final review data
    print("--- [Agent 2] Generated Review Data ---")
    return {"review_data": raw_json_output}

# %%
# --- 2.4. Async LangGraph Setup and Execution ---

async def main():
    """Main async function to run the graph."""
    print("--- LangGraph Async Custom HTTP Gemini Cascading Agent Example ---")
    print("NOTE: Requires 'langgraph' and 'aiohttp' installation.")

    # Ensure API Key is available before starting the graph
    if not os.getenv("GEMINI_API_KEY"):
        print("\nERROR: GEMINI_API_KEY environment variable not set.")
        print("Please set the GEMINI_API_KEY environment variable and try again.")
        return

    # --- Build the Graph ---
    graph_builder = StateGraph(AgentState)

    # Add the async nodes (agents) to the graph
    graph_builder.add_node("score_generator", score_generator_node)
    graph_builder.add_node("review_generator", review_generator_node)

    # Set the entry point
    graph_builder.set_entry_point("score_generator")

    # Define the sequence: Score Generator -> Review Generator -> END
    graph_builder.add_edge("score_generator", "review_generator")
    graph_builder.add_edge("review_generator", END)

    # Compile the graph into a runnable application
    app = graph_builder.compile()

    # --- Example Invocation ---
    print("\nInvoking the two-agent LangGraph application (ASYNC)...")

    # This dictionary serves as the initial state for the graph
    initial_state = {
        "item_name": "Quantum Entanglement Module v1.2",
        "complexity_level": "High/Experimental"
    }

    # Run the graph asynchronously
    final_state = await app.ainvoke(initial_state)

    final_review_json = final_state.get('review_data')
    final_score = final_state.get('score')

    print("\n--- Graph Execution Complete ---")
    print(f"Original Input:\n{json.dumps(initial_state, indent=2)}")
    print(f"Intermediate Score Generated by Agent 1: {final_score}")

    print("\n--- Final Output (Generated by Agent 2) ---")

    # Try to print the final result nicely
    try:
        print(json.dumps(json.loads(final_review_json), indent=2))
    except json.JSONDecodeError:
        print(f"Raw Output:\n{final_review_json}")

    print("\n--- End of LangGraph Execution ---")


# Run the async main function (in Jupyter, use await directly)
await main()

# %%



