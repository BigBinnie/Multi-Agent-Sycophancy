import asyncio
from typing import Dict, List, Any, Optional
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    LLMMessage,
    SystemMessage,
    UserMessage,
)


class OpenAIClient(ChatCompletionClient):
    """
    A client wrapper for OpenAI models using autogen_ext.models.openai.OpenAIChatCompletionClient.
    This provides a consistent interface with other clients in the project.
    """
    
    def __init__(
        self, 
        model: str = "gpt-4o-mini", 
        config: dict = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Initialize the OpenAI chat completion client.
        
        Args:
            model: The OpenAI model name (e.g., "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo").
            config: Configuration for the model.
            api_key: OpenAI API key. If None, will use environment variable OPENAI_API_KEY.
            base_url: Base URL for OpenAI API. If None, uses default OpenAI endpoint.
            **kwargs: Additional arguments to pass to the underlying OpenAI client.
        """
        self.model = model
        self.config = config or {}
        self._total_tokens = 0
        self._total_cost = 0
        
        # Initialize the underlying autogen OpenAI client
        client_kwargs = {}
        if api_key is not None:
            client_kwargs['api_key'] = api_key
        if base_url is not None:
            client_kwargs['base_url'] = base_url
        
        # Add any additional kwargs
        client_kwargs.update(kwargs)
        
        print(f"Initializing OpenAI client with model: {model}")
        self._client = OpenAIChatCompletionClient(model=model, **client_kwargs)
        print(f"OpenAI client initialized successfully.")

    async def create(self, messages: List[LLMMessage]) -> AssistantMessage:
        """
        Create a completion using the OpenAI model.
        
        Args:
            messages: List of messages.
            
        Returns:
            An AssistantMessage with the model's response.
        """
        # Use the underlying autogen OpenAI client directly
        response = await self._client.create(messages)
        
        # Update token usage estimation
        estimated_tokens = sum(len(msg.content) // 4 for msg in messages)
        estimated_tokens += len(response.content) // 4
        self._total_tokens += estimated_tokens
        
        return response

    async def create_stream(self, messages: List[LLMMessage]):
        """
        Create a streaming completion using the OpenAI model.
        
        Args:
            messages: List of messages.
            
        Yields:
            AssistantMessage objects with partial responses.
        """
        # Check if the underlying client supports streaming
        if hasattr(self._client, 'create_stream'):
            async for response in self._client.create_stream(messages):
                yield response
        else:
            # Fallback to non-streaming
            response = self.create(messages)
            yield response

    def count_tokens(self, messages: List[LLMMessage]) -> int:
        """
        Count the number of tokens in the messages.
        
        Args:
            messages: List of messages.
            
        Returns:
            The number of tokens (estimated).
        """
        # Simple approximation - 4 characters per token
        total = 0
        for message in messages:
            total += len(message.content) // 4
        return total

    def remaining_tokens(self, messages: List[LLMMessage]) -> int:
        """
        Calculate the remaining tokens available for the response.
        
        Args:
            messages: List of messages.
            
        Returns:
            The number of remaining tokens.
        """
        # Default context lengths for common OpenAI models
        model_limits = {
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
        }
        
        max_context_tokens = model_limits.get(self.model, 4096)
        max_tokens = self.config.get('maxTokenCount', max_context_tokens)
        used_tokens = self.count_tokens(messages)
        return max_tokens - used_tokens

    def total_usage(self) -> float:
        """
        Get the total token usage.
        
        Returns:
            The total number of tokens used.
        """
        return self._total_tokens

    def actual_usage(self) -> float:
        """
        Get the actual cost of usage.
        
        Returns:
            The total cost (estimated based on token usage).
        """
        return self._total_cost

    def capabilities(self) -> Dict[str, bool]:
        """
        Get the capabilities of the model.
        
        Returns:
            A dictionary of capabilities.
        """
        return {
            "stream": True,  # OpenAI supports streaming
            "token_count": True,
            "token_limit": True,
            "multi_turn": True
        }

    def model_info(self):
        """
        Get information about the model.
        
        Returns:
            A dictionary with model information.
        """
        # Default context lengths for common OpenAI models
        model_limits = {
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
        }
        
        max_tokens = model_limits.get(self.model, 4096)
        max_tokens = self.config.get('maxTokenCount', max_tokens)
        
        return {
            "name": self.model,
            "provider": "OpenAI",
            "max_tokens": max_tokens,
            "token_limit": max_tokens
        }

    def close(self) -> None:
        """
        Clean up resources.
        """
        if hasattr(self._client, 'close'):
            # Handle both sync and async close methods
            import asyncio
            try:
                # Try to close synchronously first
                if asyncio.iscoroutinefunction(self._client.close):
                    # If it's async, we need to handle it properly
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # If loop is running, schedule the close
                            asyncio.create_task(self._client.close())
                        else:
                            # If no loop is running, run it
                            asyncio.run(self._client.close())
                    except RuntimeError:
                        # If we can't handle async properly, just pass
                        pass
                else:
                    # Synchronous close
                    self._client.close()
            except Exception:
                # If close fails for any reason, just pass
                pass



async def openai_prediction(prompt, config, model="gpt-4o-mini", api_key=None, base_url=None):
    """
    Make a prediction using the OpenAI model via autogen_ext.models.openai.
    
    Args:
        prompt: The prompt to use (should contain 'messages' and optionally 'system_prompt').
        config: Configuration for the model.
        model: The OpenAI model name.
        api_key: OpenAI API key.
        base_url: Base URL for OpenAI API.
        
    Returns:
        The model's response in a format similar to other prediction functions.
    """
    # Initialize the OpenAI client directly from autogen
    client_kwargs = {}
    if api_key is not None:
        client_kwargs['api_key'] = api_key
    if base_url is not None:
        client_kwargs['base_url'] = base_url
    
    print("configuration", client_kwargs)
    client = OpenAIChatCompletionClient(model=model, **client_kwargs)
    
    # Format the messages
    messages = []
    
    # Add system prompt if available
    if "system_prompt" in prompt and prompt["system_prompt"]:
        messages.append(SystemMessage(content=prompt["system_prompt"]))
    
    # Add the conversation history
    for message in prompt["messages"]:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            messages.append(UserMessage(content=content, source="user"))
        elif role == "assistant":
            messages.append(AssistantMessage(content=content, source="openai"))
    
    # Generate the response
    response = await client.create(messages)
    
    # Clean up
    if hasattr(client, 'close'):
        try:
            if asyncio.iscoroutinefunction(client.close):
                await client.close()
            else:
                client.close()
        except Exception:
            pass
    
    # Return the response in a format similar to other prediction functions
    return {
        "content": [{"text": response.content}],
        "usage": {
            "total_tokens": 0  # Token usage would need to be tracked separately if needed
        }
    }
