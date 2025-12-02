import os
import pandas as pd
import requests
import json
from typing import Any, List, Optional

from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import LLMResult

from pydantic import BaseModel, Field, validator
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnableLambda

# Set your API keys
os.environ["GOOGLE_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""


# --- 1. Custom LLM Implementation ---

class CustomHTTPGemini(BaseLLM):
    """
    A custom LangChain LLM wrapper that interacts with the Google Gemini API
    using direct HTTP requests (POST to generateContent endpoint), with optional
    support for JSON output via response_schema.
    """

    # Model and API Configuration
    api_key: Optional[str] = None
    model_name: str = Field(default="gemini-2.5-flash", alias="model")
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/models/"
    response_schema: Optional[dict] = None

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if not self.api_key:
            self.api_key = os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY must be provided or set as an environment variable.")

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "custom_http_gemini"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        """
        The core logic to make the HTTP POST request to the Gemini API.
        
        Args:
            prompt: The prompt text or pre-constructed request_data JSON
            **kwargs: Can include 'request_data' to pass pre-constructed JSON
        """
        # Check if request_data is passed in kwargs
        if 'request_data' in kwargs:
            request_data = kwargs['request_data']
            
            # LINE ~47-52: Print the pre-constructed request_data
            print("\n" + "="*80)
            print("PRE-CONSTRUCTED REQUEST_DATA JSON:")
            print("="*80)
            print(json.dumps(request_data, indent=2))
            print("="*80 + "\n")
        else:
            # Fallback: construct request_data from prompt string
            request_data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ]
            }
            
            # Add generation configuration for JSON output if schema is present
            if self.response_schema:
                request_data["generationConfig"] = {
                    "responseMimeType": "application/json",
                    "responseSchema": self.response_schema
                }
            
            # LINE ~76-81: Print the final rendered prompt (fallback mode)
            print("\n" + "="*80)
            print("FINAL RENDERED PROMPT BEING SENT TO GEMINI API:")
            print("="*80)
            print(prompt)
            print("="*80 + "\n")
        
        api_endpoint = f"{self.base_url}{self.model_name}:generateContent"
        url = f"{api_endpoint}?key={self.api_key}"
        headers = {"Content-Type": "application/json"}

        request_data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
        }

        if self.response_schema:
            request_data["generationConfig"] = {
                "responseMimeType": "application/json",
                "responseSchema": self.response_schema
            }

        try:
            response = requests.post(url=url, headers=headers, json=request_data)
            response.raise_for_status()
            response_json = response.json()
            generated_text = response_json['candidates'][0]['content']['parts'][0]['text']
            return generated_text

        except requests.exceptions.HTTPError as err:
            error_message = f"Gemini API HTTP Error ({err.response.status_code}): {err.response.text}"
            raise RuntimeError(error_message) from err
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred during API call: {e}")

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Call the LLM on a list of prompts."""
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop, run_manager, **kwargs)
            generations.append([{"text": text}])
        return LLMResult(generations=generations)


# --- 2. Custom Wrapper to Construct and Pass request_data ---

class RequestDataBuilder:
    """
    Helper class to construct request_data JSON before passing to LLM.
    """
    
    @staticmethod
    def build_request_data(prompt: str, response_schema: Optional[dict] = None) -> dict:
        """
        Construct the request_data JSON structure.
        
        Args:
            prompt: The text prompt to send
            response_schema: Optional JSON schema for structured output
            
        Returns:
            Complete request_data dictionary
        """
        request_data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
        }
        
        # Add generation configuration for JSON output if schema is present
        if response_schema:
            request_data["generationConfig"] = {
                "responseMimeType": "application/json",
                "responseSchema": response_schema
            }
        
        return request_data


# Custom LLM wrapper that uses request_data
class CustomHTTPGeminiWithRequestData(BaseLLM):
    """
    Extended version that accepts pre-constructed request_data.
    """
    
    api_key: Optional[str] = None
    model_name: str = Field(default="gemini-2.5-flash", alias="model")
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/models/"
    response_schema: Optional[dict] = None

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if not self.api_key:
            self.api_key = os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY must be provided or set as an environment variable.")

    @property
    def _llm_type(self) -> str:
        return "custom_http_gemini_with_request_data"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        """
        Accept either a prompt string or pre-constructed request_data.
        """
        # Check if request_data is passed in kwargs
        if 'request_data' in kwargs:
            request_data = kwargs['request_data']
            
            # LINE ~157-162: Print the pre-constructed request_data
            print("\n" + "="*80)
            print("PRE-CONSTRUCTED REQUEST_DATA JSON:")
            print("="*80)
            print(json.dumps(request_data, indent=2))
            print("="*80 + "\n")
        else:
            # Fallback: construct request_data from prompt string
            request_data = RequestDataBuilder.build_request_data(prompt, self.response_schema)
            
            # LINE ~167-172: Print the prompt (fallback mode)
            print("\n" + "="*80)
            print("FINAL RENDERED PROMPT (FALLBACK MODE):")
            print("="*80)
            print(prompt)
            print("="*80 + "\n")
        
        api_endpoint = f"{self.base_url}{self.model_name}:generateContent"
        url = f"{api_endpoint}?key={self.api_key}"
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(url=url, headers=headers, json=request_data)
            response.raise_for_status()
            response_json = response.json()
            generated_text = response_json['candidates'][0]['content']['parts'][0]['text']
            return generated_text

        except requests.exceptions.HTTPError as err:
            error_message = f"Gemini API HTTP Error ({err.response.status_code}): {err.response.text}"
            raise RuntimeError(error_message) from err
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred during API call: {e}")

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop, run_manager, **kwargs)
            generations.append([{"text": text}])
        return LLMResult(generations=generations)


# --- 3. JSON Schema Definitions ---

SCORE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "score": {"type": "NUMBER", "description": "A random technical score between 0.0 and 1.0."},
    },
    "required": ["score"],
    "propertyOrdering": ["score"]
}

REVIEW_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "review_text": {"type": "STRING", "description": "A concise, technical review."},
        "category": {"type": "STRING", "description": "The category of the review (e.g., 'Positive', 'Neutral', 'Negative')."}
    },
    "required": ["review_text", "category"],
    "propertyOrdering": ["review_text", "category"]
}


# --- 4. CSV Processing Function ---

def process_csv_with_cascading_agents(csv_path: str, user_prompt: str = "", output_path: str = "results.csv"):
    """
    Read a CSV file and process each record through the cascading agent chain.
    
    Args:
        csv_path: Path to the input CSV file
        user_prompt: Custom user prompt to be included in each agent invocation
        output_path: Path to save the results CSV
    
    Expected CSV columns: 'item_name', 'complexity_level'
    """
    
    print(f"--- Reading CSV from: {csv_path} ---")
    print(f"--- User Prompt: {user_prompt} ---" if user_prompt else "--- No custom user prompt provided ---")
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} records from CSV")
    except FileNotFoundError:
        print(f"ERROR: CSV file not found at {csv_path}")
        return
    except Exception as e:
        print(f"ERROR reading CSV: {e}")
        return
    
    # Validate required columns
    required_columns = ['item_name', 'complexity_level']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"ERROR: Missing required columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Initialize LLM instances with the new class
    try:
        llm_score_generator = CustomHTTPGeminiWithRequestData(model_name="gemini-2.5-flash", response_schema=SCORE_SCHEMA)
        llm_review_generator = CustomHTTPGeminiWithRequestData(model_name="gemini-2.5-flash", response_schema=REVIEW_SCHEMA)
    except ValueError as e:
        print(f"\nERROR: {e}")
        print("Please set the GEMINI_API_KEY environment variable and try again.")
        return
    
    # --- Build the Cascading Chain ---
    
    # AGENT 1: Score Generator
    prompt_1_template = """
{user_prompt}

You are a technical analyst. Your task is to assign a random score between 0.0 and 1.0 to the '{item_name}' based on its complexity '{complexity_level}'. Output the result strictly in JSON format according to the schema.
""".strip()
    
    prompt_1 = PromptTemplate.from_template(prompt_1_template)
    
    # Custom function to construct request_data and pass it to LLM
    def invoke_agent_1_with_request_data(input_dict: dict) -> str:
        # Render the prompt template
        rendered_prompt = prompt_1.format(**input_dict)
        
        # LINE ~275-280: Construct request_data outside _call
        print("\n" + "^"*80)
        print("CONSTRUCTING REQUEST_DATA FOR AGENT 1 (outside _call):")
        print("^"*80)
        request_data = RequestDataBuilder.build_request_data(rendered_prompt, SCORE_SCHEMA)
        print(json.dumps(request_data, indent=2))
        print("^"*80 + "\n")
        
        # Pass request_data to the LLM via kwargs
        return llm_score_generator._call(rendered_prompt, request_data=request_data)
    
    chain_1 = RunnableLambda(invoke_agent_1_with_request_data) | StrOutputParser()
    
    # Mapping Step
    def map_score_to_review_context(input_dict: dict) -> dict:
        score_json_str = input_dict.pop("output")
        
        try:
            score_data = json.loads(score_json_str)
            score = score_data.get('score', 0.0)
            
            result_dict = {
                "user_prompt": input_dict["user_prompt"],
                "item_name": input_dict["item_name"],
                "score": score,
                "complexity_level": input_dict["complexity_level"],
            }
            
            # LINE ~146-155: Print mapped context for Agent 2
            print(f"\n{'-'*80}")
            print("MAPPED CONTEXT FOR AGENT 2 (Review Generator):")
            print('-'*80)
            print(f"User Prompt: {result_dict['user_prompt']}")
            print(f"Item Name: {result_dict['item_name']}")
            print(f"Score (from Agent 1): {result_dict['score']}")
            print(f"Complexity Level: {result_dict['complexity_level']}")
            print('-'*80)
            
            # LINE ~157-161: Print Agent 2 prompt template
            print(f"\n{'*'*80}")
            print("PROMPT TEMPLATE FOR AGENT 2 (Review Generator):")
            print('*'*80)
            print(prompt_2_template)
            print('*'*80)
            
            return result_dict
            
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from Agent 1: {e}")
            return {
                "user_prompt": input_dict["user_prompt"],
                "item_name": input_dict["item_name"],
                "score": 0.5,
                "complexity_level": input_dict["complexity_level"],
            }
    
    score_with_context_chain = {
        "user_prompt": RunnableLambda(lambda x: x["user_prompt"]),
        "item_name": RunnableLambda(lambda x: x["item_name"]),
        "complexity_level": RunnableLambda(lambda x: x["complexity_level"]),
        "output": chain_1
    }
    
    score_mapper = score_with_context_chain | RunnableLambda(map_score_to_review_context)
    
    # AGENT 2: Review Generator
    prompt_2_template = """
{user_prompt}

Generate a technical review for the item '{item_name}' which has a complexity of '{complexity_level}' and received a technical score of {score}. Your review must reflect this score. Output the review strictly in JSON format according to the schema.
""".strip()
    
    prompt_2 = PromptTemplate.from_template(prompt_2_template)
    
    # Custom function to construct request_data and pass it to LLM
    def invoke_agent_2_with_request_data(input_dict: dict) -> str:
        # Render the prompt template
        rendered_prompt = prompt_2.format(**input_dict)
        
        # LINE ~330-335: Construct request_data outside _call
        print("\n" + "^"*80)
        print("CONSTRUCTING REQUEST_DATA FOR AGENT 2 (outside _call):")
        print("^"*80)
        request_data = RequestDataBuilder.build_request_data(rendered_prompt, REVIEW_SCHEMA)
        print(json.dumps(request_data, indent=2))
        print("^"*80 + "\n")
        
        # Pass request_data to the LLM via kwargs
        return llm_review_generator._call(rendered_prompt, request_data=request_data)
    
    chain_2 = RunnableLambda(invoke_agent_2_with_request_data) | StrOutputParser()
    
    # Final Cascading Chain
    final_chain = score_mapper | chain_2
    
    # --- Process Each Record ---
    
    results = []
    
    for idx, row in df.iterrows():
        # LINE ~179-182: Print record header
        print(f"\n{'#'*80}")
        print(f"### Processing Record {idx + 1}/{len(df)} ###")
        print(f"{'#'*80}")
        
        # Prepare input for the chain
        input_data = {
            "user_prompt": user_prompt,
            "item_name": str(row['item_name']),
            "complexity_level": str(row['complexity_level'])
        }
        
        # LINE ~193-200: Print input record values
        print(f"\n{'~'*80}")
        print("INPUT RECORD VALUES:")
        print('~'*80)
        print(f"User Prompt: {input_data['user_prompt']}")
        print(f"Item Name: {input_data['item_name']}")
        print(f"Complexity Level: {input_data['complexity_level']}")
        print('~'*80)
        
        # LINE ~202-206: Print Agent 1 prompt template
        print(f"\n{'*'*80}")
        print("PROMPT TEMPLATE FOR AGENT 1 (Score Generator):")
        print('*'*80)
        print(prompt_1_template)
        print('*'*80)
        
        try:
            # Invoke the cascading chain
            response = final_chain.invoke(input_data)
            
            # Parse the JSON response
            review_data = json.loads(response)
            
            # Store results
            result = {
                "item_name": input_data["item_name"],
                "complexity_level": input_data["complexity_level"],
                "review_text": review_data.get("review_text", ""),
                "category": review_data.get("category", ""),
            }
            
            results.append(result)
            print(f"Success: {result['category']} - {result['review_text'][:50]}...")
            
        except Exception as e:
            print(f"ERROR processing record {idx + 1}: {e}")
            # Add error record
            results.append({
                "item_name": input_data["item_name"],
                "complexity_level": input_data["complexity_level"],
                "review_text": f"ERROR: {str(e)}",
                "category": "Error",
            })
    
    # --- Save Results to CSV ---
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"\n--- Results saved to: {output_path} ---")
    print(f"Total records processed: {len(results)}")
    
    return results_df


if __name__ == "__main__":
    print("--- LangChain Custom HTTP Gemini Cascading Agent with CSV Processing ---")
    
    # Example: Create a sample CSV file if it doesn't exist
    sample_csv = "sample_items.csv"
    
    if not os.path.exists(sample_csv):
        print(f"\nCreating sample CSV file: {sample_csv}")
        sample_data = pd.DataFrame({
            "item_name": [
                "Quantum Entanglement Module v1.2",
                "Neural Network Processor X500",
                "Blockchain Validator Node Pro",
                "AI Ethics Framework Standard",
                "Distributed Ledger Protocol v3"
            ],
            "complexity_level": [
                "High/Experimental",
                "Medium/Production",
                "High/Enterprise",
                "Low/Conceptual",
                "Medium/Research"
            ]
        })
        sample_data.to_csv(sample_csv, index=False)
        print(f"Sample CSV created with {len(sample_data)} records")
    
    # Process the CSV file
    csv_file_path = sample_csv  # Change this to your CSV file path
    output_file_path = "review_results.csv"
    
    # Define your custom user prompt
    custom_user_prompt = """
Please consider the following context when performing your analysis:
- Focus on technical accuracy and innovation potential
- Consider market readiness and adoption barriers
- Evaluate long-term sustainability and scalability
"""
    
    results = process_csv_with_cascading_agents(
        csv_path=csv_file_path,
        user_prompt=custom_user_prompt,
        output_path=output_file_path
    )
    
    if results is not None:
        print("\n--- Sample Results ---")
        print(results.head())
