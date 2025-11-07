import boto3
import json
import time
import random
from typing import Dict, List, Any
from omegaconf import OmegaConf
from config_schema import ModelConfig
from botocore.exceptions import ClientError

from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    LLMMessage,
    SystemMessage,
    UserMessage,
)

def to_dict(obj):
    """Convert OmegaConf objects to standard Python types"""
    if hasattr(obj, '_is_config') or OmegaConf.is_config(obj):
        return OmegaConf.to_container(obj, resolve=True)
    return obj

def invoke_with_retry(bedrock, body, model_id, max_retries=5, initial_backoff=1, max_backoff=60):
    """
    Invoke the Bedrock model with exponential backoff retry for ThrottlingException
    
    Args:
        bedrock: Bedrock client
        body: Request body
        model_id: Model ID
        max_retries: Maximum number of retries
        initial_backoff: Initial backoff time in seconds
        max_backoff: Maximum backoff time in seconds
        
    Returns:
        Response from the model
    """
    retries = 0
    backoff = initial_backoff
    
    while True:
        try:
            response = bedrock.invoke_model(body=body, modelId=model_id)
            return response
        except ClientError as e:
            if e.response['Error']['Code'] == 'ThrottlingException' and retries < max_retries:
                # Add jitter to avoid thundering herd problem
                jitter = random.uniform(0, 0.1 * backoff)
                sleep_time = backoff + jitter
                
                print(f"ThrottlingException encountered. Retrying in {sleep_time:.2f} seconds (retry {retries+1}/{max_retries})")
                time.sleep(sleep_time)
                
                # Exponential backoff with jitter
                backoff = min(backoff * 2, max_backoff)
                retries += 1
            else:
                # Re-raise the exception if it's not a ThrottlingException or we've exceeded max retries
                raise

def claude3_prediction(bedrock, prompt, config, model_id):
    # Convert OmegaConf objects to standard Python types
    config = to_dict(config)
    
    # Extract stop sequences
    stop_sequences = config.get("stopSequences", [])
    if hasattr(stop_sequences, '_is_config') or OmegaConf.is_config(stop_sequences):
        stop_sequences = OmegaConf.to_container(stop_sequences, resolve=True)
    
    json_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": config.get("maxTokenCount", 2048),
        "messages": prompt["messages"],
        "stop_sequences": stop_sequences,
        "temperature": config.get("temperature", 1),
        "top_p": 1
    }
    
    if "system_prompt" in prompt:
        print("the system prompt is: ", prompt["system_prompt"] )
        json_body["system"] = prompt["system_prompt"]
    
    print("configuration", json_body)
    
    body = json.dumps(json_body)  
    response = invoke_with_retry(bedrock, body, model_id)
    response_body = json.loads(response.get('body').read())
   
    return response_body

def llama_prediction(bedrock, prompt, config, model_id):
    # Convert OmegaConf objects to standard Python types
    config = to_dict(config)
    
    # Format the chat messages into a prompt string for Llama models
    formatted_prompt = ""
    
    # Add system prompt if available
    if "system_prompt" in prompt and prompt["system_prompt"]:
        formatted_prompt += f"<s>[INST] <<SYS>>\n{prompt['system_prompt']}\n<</SYS>>\n\n"
    else:
        formatted_prompt += "<s>[INST] "
    
    # Add the conversation history
    for i, message in enumerate(prompt["messages"]):
        role = message["role"]
        content = message["content"]
        
        if i == 0 and role == "user":
            # First user message goes with the system prompt
            formatted_prompt += f"{content} [/INST]"
        elif role == "user":
            formatted_prompt += f"</s><s>[INST] {content} [/INST]"
        elif role == "assistant":
            formatted_prompt += f" {content} "
    
    # Prepare the request body
    json_body = {
        "prompt": formatted_prompt,
        "max_gen_len": config.get("maxTokenCount", 2048),
        "temperature": config.get("temperature", 0.7),
        "top_p": 1
    }
    
    print("Llama configuration:", json_body)
    
    body = json.dumps(json_body)
    response = invoke_with_retry(bedrock, body, model_id)
    response_body = json.loads(response.get('body').read())
    print("response: ", response_body)
    return response_body

def deepseek_r1_prediction(bedrock, prompt, config, model_id):
    # Convert OmegaConf objects to standard Python types
    config = to_dict(config)
    
    # Extract stop sequences
    stop_sequences = config.get("stopSequences", [])
    if hasattr(stop_sequences, '_is_config') or OmegaConf.is_config(stop_sequences):
        stop_sequences = OmegaConf.to_container(stop_sequences, resolve=True)
    
    # Format the prompt for Deepseek R1 model
    formatted_prompt = ""
    
    # Add system prompt if available
    if "system_prompt" in prompt and prompt["system_prompt"]:
        formatted_prompt += f"System: {prompt['system_prompt']}\n\n"
    
    # Add the conversation history
    for i, message in enumerate(prompt["messages"]):
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            formatted_prompt += f"Human: {content}\n\n"
        elif role == "assistant":
            formatted_prompt += f"Assistant: {content}\n\n"
    
    formatted_prompt += "List the steps, keeping each step under 20 words.\n\n"
    # Ensure the prompt ends with "Assistant: " to indicate where the model should continue
    if not formatted_prompt.endswith("Assistant: "):
        formatted_prompt += "Assistant: "
    
    # Prepare the request body
    json_body = {
        "prompt": formatted_prompt,
        "max_tokens": config.get("maxTokenCount", 2048),
        "temperature": config.get("temperature", 1),
        "top_p": 1,
        "stop": stop_sequences,
    }
    
    print("Deepseek R1 configuration:", json_body)
    
    body = json.dumps(json_body)
    response = invoke_with_retry(bedrock, body, model_id)
    response_body = json.loads(response.get('body').read())
    print("response: ", response_body)
    return response_body

class BedrockChatCompletionClient(ChatCompletionClient):
    def __init__(self, model_id: str, config: dict, region_name: str = 'us-east-1') -> None:
        self.bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name=region_name
        )
        self.model_id = model_id
        self.config = config
        self._total_tokens = 0
        self._total_cost = 0

    async def create(self, messages: List[LLMMessage]) -> AssistantMessage:
        formatted_messages = []
        system_prompt = ""
        
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_prompt = msg.content
            else:
                formatted_messages.append({
                    "role": "user" if isinstance(msg, UserMessage) else "assistant",
                    "content": msg.content
                })

        prompt = {
            "system_prompt": system_prompt,
            "messages": formatted_messages
        }

        # Ensure config is a standard Python dict
        config = to_dict(self.config)

        # Choose the appropriate prediction function based on the model ID
        if "llama" in self.model_id.lower():
            response = llama_prediction(self.bedrock, prompt, config, self.model_id)
            # Extract content from Llama response and remove special tokens
            content = response.get('generation', '')
            # Remove special tokens that might be in the response
            content = content.split('[/SYS]')[0].strip()
            content = content.replace('<s>', '').replace('</s>', '').replace('[INST]', '').replace('[/INST]', '').replace('<<SYS>>', '').replace('<</SYS>>','').strip()
        elif "deepseek" in self.model_id.lower():
            response = deepseek_r1_prediction(self.bedrock, prompt, config, self.model_id)
            # Extract content from Deepseek R1 response
            content = response['choices'][0]['text']
            content = content.split('Human:')[0].strip()
        else:
            response = claude3_prediction(self.bedrock, prompt, config, self.model_id)
            # Extract content from Claude response
            content = response['content'][0]['text'].strip()
        # Update token usage (if available in response)
        if 'usage' in response:
            self._total_tokens += response['usage'].get('total_tokens', 0)

        return AssistantMessage(content=content, source="bedrock")

    def create_sync(self, messages: List[LLMMessage]) -> AssistantMessage:
        """Synchronous version of create method."""
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.create(messages))
        finally:
            loop.close()

    def create_batch_sync(self, batch_messages: List[List[LLMMessage]]) -> List[AssistantMessage]:
        """Create batch completions synchronously."""
        responses = []
        for messages in batch_messages:
            response = self.create_sync(messages)
            responses.append(response)
        return responses

    async def create_batch(self, batch_messages: List[List[LLMMessage]]) -> List[AssistantMessage]:
        """Create batch completions asynchronously."""
        responses = []
        for messages in batch_messages:
            response = await self.create(messages)
            responses.append(response)
        return responses

    async def create_stream(self, messages: List[LLMMessage]):
        # Implement streaming if needed
        response = await self.create(messages)
        yield response

    def count_tokens(self, messages: List[LLMMessage]) -> int:
        # Simple approximation - you might want to implement a more accurate token counter
        total = 0
        for message in messages:
            # Rough estimate: 4 characters per token
            total += len(message.content) // 4
        return total

    def remaining_tokens(self, messages: List[LLMMessage]) -> int:
        # For Bedrock, this could be the model's maximum context length minus used tokens
        config = to_dict(self.config)
            
        max_tokens = config.get('maxTokenCount', 2048)
        used_tokens = self.count_tokens(messages)
        return max_tokens - used_tokens

    def total_usage(self) -> float:
        return self._total_tokens

    def actual_usage(self) -> float:
        return self._total_cost

    def capabilities(self) -> Dict[str, bool]:
        return {
            "stream": False,  # Set to True if streaming is implemented
            "token_count": True,
            "token_limit": True,
            "multi_turn": True
        }

    def model_info(self):
        # Ensure config is a standard Python dict for accessing values
        config = to_dict(self.config)
            
        return {
            "name": self.model_id,
            "provider": "Amazon Bedrock",
            "max_tokens": config.get('maxTokenCount', 2048),
            "token_limit": config.get('maxTokenCount', 2048)
        }

    def close(self) -> None:
        pass
