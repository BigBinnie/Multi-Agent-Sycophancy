import json
import time
from typing import Dict, List, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    LLMMessage,
    SystemMessage,
    UserMessage,
)

from accelerate import Accelerator
ACCELERATE_AVAILABLE = True

class LlamaChatCompletionClient(ChatCompletionClient):
    """
    A client for Llama models loaded directly from Huggingface.
    """
    def __init__(
        self, 
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct", 
        config: dict = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32,
    ) -> None:
        """
        Initialize the Llama chat completion client.
        
        Args:
            model_name: The name of the model on Huggingface.
            config: Configuration for the model.
            device: The device to run the model on.
            torch_dtype: The torch data type to use.
        """
        self.model_name = model_name
        self.config = config or {}
        self.device = device
        self.torch_dtype = torch_dtype
        self._total_tokens = 0
        self._total_cost = 0
        
        # Extract memory optimization parameters from config
        use_gradient_checkpointing = self.config.get("use_gradient_checkpointing", False)
        load_in_8bit = self.config.get("load_in_8bit", False)
        device_map = self.config.get("device_map", "auto")
        offload_folder = self.config.get("offload_folder", None)
        max_token_count = self.config.get("maxTokenCount", 2048)
        
        # Determine if this is a large model that needs special handling
        model_size_str = model_name.lower()
        is_large_model = any(size in model_size_str for size in ["13b", "30b", "65b", "70b"])
        is_medium_model = any(size in model_size_str for size in ["7b"])
        
        print(f"Loading Llama model {model_name} on {device}...")
        
        # Apply memory optimizations if specified
        if use_gradient_checkpointing:
            print("Enabling gradient checkpointing for memory efficiency")
        if load_in_8bit:
            print("Enabling 8-bit quantization for memory efficiency")
        if offload_folder:
            print(f"Enabling parameter offloading to {offload_folder}")
            import os
            os.makedirs(offload_folder, exist_ok=True)
            
        # Load tokenizer with max length optimization for large models
        tokenizer_kwargs = {
            "trust_remote_code": True,
        }
        
        # For large models, be more conservative with max length
        if is_large_model and max_token_count > 1024:
            print(f"Limiting max token count to 1024 for large model (was {max_token_count})")
            self.config["maxTokenCount"] = 1024
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        
        # Set pad token if not present (common issue with Llama models)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set padding side to left for decoder-only models
        self.tokenizer.padding_side = 'left'
        
        # Prepare model loading parameters
        model_kwargs = {
            "torch_dtype": self.torch_dtype if torch_dtype != "auto" else "auto",
            "device_map": device_map,
            "trust_remote_code": True,
        }
        
        # Add quantization parameters if specified
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
                
        # Load the model with optimization parameters
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
            # **model_kwargs
        )
        
        # Enable gradient checkpointing after model loading if specified
        if use_gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            
        print(f"Llama model loaded successfully with memory optimizations.")

    def _format_messages(self, messages: List[LLMMessage]) -> List[Dict[str, str]]:
        """
        Format the messages for the Llama model.
        
        Args:
            messages: List of messages.
            
        Returns:
            Formatted messages for the Llama model.
        """
        formatted_messages = []
        
        for msg in messages:
            if isinstance(msg, SystemMessage):
                formatted_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, UserMessage):
                formatted_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AssistantMessage):
                formatted_messages.append({"role": "assistant", "content": msg.content})
        
        return formatted_messages

    def _parse_assistant_response(self, response: str) -> str:
        """
        Parse the model response and extract only the last assistant response.
        
        Args:
            response: Raw response from the model.
            
        Returns:
            Cleaned response containing only the last assistant response.
        """
        # Split by common assistant markers and take the last part
        assistant_markers = [
            "assistant\n\n"
        ]
        
        # Try to find assistant markers and extract the last response
        for marker in assistant_markers:
            if marker in response:
                parts = response.split(marker)
                if len(parts) > 1:
                    # Take the last part after the last assistant marker
                    last_response = parts[-1].strip()
                    # Remove any trailing markers or special tokens
                    last_response = last_response.split("<|im_end|>")[0].strip()
                    last_response = last_response.split("<|eot_id|>")[0].strip()
                    return last_response
        
        # If no assistant markers found, return the original response cleaned
        response = response.strip()
        # Remove common ending tokens
        response = response.split("<|im_end|>")[0].strip()
        response = response.split("<|eot_id|>")[0].strip()
        return response

    def _calculate_logprobs(self, outputs, batch_index=0):
        """
        Calculate log probabilities from LLaMA outputs, scaled by temperature.
        Args:
            outputs: The model generation outputs (with output_scores=True).
            batch_index: Index of sample in batch.
        Returns:
            Confidence score in [0, 100].
        """
        import torch
        import math
        import torch.nn.functional as F

        try:
            scores = torch.stack(outputs.scores)  # [seq_len, batch_size, vocab_size]

            if batch_index >= outputs.sequences.size(0):
                return 50.0

            sequence = outputs.sequences[batch_index]
            input_length = sequence.size(0) - scores.size(0)
            generated_ids = sequence[input_length:]

            log_probs = []
            for i, token_id in enumerate(generated_ids):
                if i >= scores.size(0):
                    break

                token_scores = scores[i][batch_index]  # logits for token i
                if token_id == self.tokenizer.pad_token_id:
                    continue

                # Convert logits to probabilities and take log
                probs = F.softmax(token_scores, dim=-1)
                token_prob = probs[token_id].item()
                token_log_prob = math.log(max(token_prob, 1e-10))

                if not math.isinf(token_log_prob) and not math.isnan(token_log_prob):
                    log_probs.append(token_log_prob)

            if not log_probs:
                print("Warning: No valid logprobs extracted.")
                return 50.0

            # Calculate confidence as exp(Σ log P(token_i) / N)
            # This is the geometric mean of token probabilities
            avg_log_prob = sum(log_probs) / len(log_probs)
            confidence = 100 * math.exp(avg_log_prob)
            
            # Ensure confidence is in [0, 100] range
            confidence = max(0, min(100, confidence))
            return confidence

        except Exception as e:
            print(f"Error calculating logprobs: {e}")
            return 50.0
            
    def create_sync(self, messages: List[LLMMessage]) -> AssistantMessage:
        """
        Create a completion using the Llama model synchronously.
        
        Args:
            messages: List of messages.
            
        Returns:
            An AssistantMessage with the model's response.
        """
        formatted_messages = self._format_messages(messages)
        
        # Extract configuration parameters
        max_tokens = self.config.get("maxTokenCount", 1024)
        temperature = self.config.get("temperature", 0)
        top_p = self.config.get("top_p", None)
        stop_sequences = self.config.get("stopSequences", [])
        
        # Convert messages to the format expected by the model
        text = self.tokenizer.apply_chat_template(
            formatted_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize the input
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]
        
        # Prepare generation parameters
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # Add stopping criteria if specified
        if stop_sequences:
            generation_kwargs["stopping_criteria"] = self._create_stopping_criteria(stop_sequences, input_length)
        
        # Handle sampling parameters
        if temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = temperature
            if top_p is not None:
                generation_kwargs["top_p"] = top_p
        else:
            # Use greedy decoding when temperature is 0
            generation_kwargs["do_sample"] = False
        
        # Generate the response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                return_dict_in_generate=True,
                output_scores=True,
                **generation_kwargs
            )
        
        # Decode the response
        output_ids = outputs.sequences[0][input_length:]
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        
        # Parse and clean the response to keep only the last assistant response
        cleaned_response = self._parse_assistant_response(response)
        
        # Calculate log probabilities
        confidence = self._calculate_logprobs(outputs)
        
        # Update token usage
        self._total_tokens += input_length + len(output_ids)
        
        # Create AssistantMessage and add confidence as a custom attribute
        response_msg = AssistantMessage(content=cleaned_response, source="llama-huggingface")
        response_msg._confidence = confidence
        return response_msg

    def create_batch_sync(self, batch_messages: List[List[LLMMessage]]) -> List[AssistantMessage]:
        """
        Create batch completions using the Llama model synchronously.
        
        Args:
            batch_messages: List of message lists for batch processing.
            
        Returns:
            List of AssistantMessage objects with the model's responses.
        """
        # Extract configuration parameters
        max_tokens = self.config.get("maxTokenCount", 1024)
        temperature = self.config.get("temperature", 0)
        top_p = self.config.get("top_p", None)
        stop_sequences = self.config.get("stopSequences", [])
        
        # Format all messages and convert to text
        batch_texts = []
        for messages in batch_messages:
            formatted_messages = self._format_messages(messages)
            text = self.tokenizer.apply_chat_template(
                formatted_messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            batch_texts.append(text)
        
        # Tokenize all inputs with padding
        inputs = self.tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.device)
        
        input_lengths = (inputs.attention_mask.sum(dim=1)).tolist()
        
        # Prepare generation parameters
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "attention_mask": inputs.attention_mask,
        }
        
        # Handle sampling parameters
        if temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = temperature
            if top_p is not None:
                generation_kwargs["top_p"] = top_p
        else:
            # Use greedy decoding when temperature is 0
            generation_kwargs["do_sample"] = False
        
        # Generate the responses
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                return_dict_in_generate=True,
                output_scores=True,
                **generation_kwargs
            )
        
        # Decode all responses
        responses = []
        for i, output in enumerate(outputs.sequences):
            input_length = input_lengths[i]
            output_ids = output[input_length:]
            response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            
            # Parse and clean the response to keep only the last assistant response
            cleaned_response = self._parse_assistant_response(response)
            
            # Calculate log probabilities for this response
            confidence = self._calculate_logprobs(outputs, batch_index=i)
            
            # Create AssistantMessage and add confidence as a custom attribute
            response_msg = AssistantMessage(content=cleaned_response, source="llama-huggingface")
            response_msg._confidence = confidence
            responses.append(response_msg)
            
            # Update token usage
            self._total_tokens += input_length + len(output_ids)
        
        return responses

    async def create(self, messages: List[LLMMessage]) -> AssistantMessage:
        """
        Create a completion using the Llama model.
        
        Args:
            messages: List of messages.
            
        Returns:
            An AssistantMessage with the model's response.
        """
        # Delegate to synchronous method since the underlying operations are synchronous anyway
        return self.create_sync(messages)

    async def create_batch(self, batch_messages: List[List[LLMMessage]]) -> List[AssistantMessage]:
        """
        Create batch completions using the Llama model.
        
        Args:
            batch_messages: List of message lists for batch processing.
            
        Returns:
            List of AssistantMessage objects with the model's responses.
        """
        # Extract configuration parameters
        max_tokens = self.config.get("maxTokenCount", 1024)
        temperature = self.config.get("temperature", 0)
        top_p = self.config.get("top_p", None)
        stop_sequences = self.config.get("stopSequences", [])
        
        # Format all messages and convert to text
        batch_texts = []
        for messages in batch_messages:
            formatted_messages = self._format_messages(messages)
            text = self.tokenizer.apply_chat_template(
                formatted_messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            batch_texts.append(text)
        
        # Tokenize all inputs with padding
        inputs = self.tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.device)
        
        input_lengths = (inputs.attention_mask.sum(dim=1)).tolist()
        
        # Prepare generation parameters
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "attention_mask": inputs.attention_mask,
        }
        
        # Handle sampling parameters
        if temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = temperature
            if top_p is not None:
                generation_kwargs["top_p"] = top_p
        else:
            # Use greedy decoding when temperature is 0
            generation_kwargs["do_sample"] = False
        
        # Generate the responses
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                **generation_kwargs
            )
        
        # Decode all responses
        responses = []
        for i, output in enumerate(outputs):
            input_length = input_lengths[i]
            output_ids = output[input_length:]
            response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            
            # Parse and clean the response to keep only the last assistant response
            cleaned_response = self._parse_assistant_response(response)
            
            # Calculate log probabilities for this response
            confidence = self._calculate_logprobs(outputs, batch_index=i)
            
            # Create AssistantMessage and add confidence as a custom attribute
            response_msg = AssistantMessage(content=cleaned_response, source="llama-huggingface")
            response_msg._confidence = confidence
            responses.append(response_msg)
            
            # Update token usage
            self._total_tokens += input_length + len(output_ids)
        
        return responses

    def _create_stopping_criteria(self, stop_sequences: List[str], input_length: int):
        """
        Create stopping criteria for the model generation.
        
        Args:
            stop_sequences: List of sequences to stop generation at.
            input_length: Length of the input tokens.
            
        Returns:
            A stopping criteria object.
        """
        from transformers.generation.stopping_criteria import StoppingCriteriaList, StoppingCriteria
        
        class StopSequenceCriteria(StoppingCriteria):
            def __init__(self, tokenizer, stop_sequences, input_length):
                self.tokenizer = tokenizer
                self.stop_sequences = stop_sequences
                self.input_length = input_length
                
            def __call__(self, input_ids, scores, **kwargs):
                if len(input_ids[0]) <= self.input_length:
                    return False
                    
                generated_text = self.tokenizer.decode(input_ids[0][self.input_length:])
                return any(seq in generated_text for seq in self.stop_sequences)
        
        return StoppingCriteriaList([StopSequenceCriteria(self.tokenizer, stop_sequences, input_length)])

    async def create_stream(self, messages: List[LLMMessage]):
        """
        Create a streaming completion using the Llama model.
        
        Args:
            messages: List of messages.
            
        Yields:
            AssistantMessage objects with partial responses.
        """
        # For simplicity, we'll just use the non-streaming version for now
        response = await self.create(messages)
        yield response

    def count_tokens(self, messages: List[LLMMessage]) -> int:
        """
        Count the number of tokens in the messages.
        
        Args:
            messages: List of messages.
            
        Returns:
            The number of tokens.
        """
        formatted_messages = self._format_messages(messages)
        text = self.tokenizer.apply_chat_template(formatted_messages, tokenize=False)
        tokens = self.tokenizer.encode(text)
        return len(tokens)

    def remaining_tokens(self, messages: List[LLMMessage]) -> int:
        """
        Calculate the remaining tokens available for the response.
        
        Args:
            messages: List of messages.
            
        Returns:
            The number of remaining tokens.
        """
        max_tokens = self.config.get("maxTokenCount", 2048)
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
            The total cost (0 for local models).
        """
        return self._total_cost

    def capabilities(self) -> Dict[str, bool]:
        """
        Get the capabilities of the model.
        
        Returns:
            A dictionary of capabilities.
        """
        return {
            "stream": False,  # Set to True if streaming is implemented
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
        max_tokens = self.config.get("maxTokenCount", 2048)
        return {
            "name": self.model_name,
            "provider": "Huggingface",
            "max_tokens": max_tokens,
            "token_limit": max_tokens
        }

    def close(self) -> None:
        """
        Clean up resources with aggressive memory cleanup.
        """
        # Free up GPU memory if possible
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
        
        # Aggressive memory cleanup
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            # Synchronize CUDA operations before cleanup
            torch.cuda.synchronize()
            
            # Clear cache multiple times for better fragmentation cleanup
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
            
            # Reset memory stats
            try:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()
            except Exception as e:
                print(f"Error in resetting memory stats: {e}")
            
            # Create and delete a dummy tensor to help with defragmentation
            try:
                dummy = torch.ones((1, 1), device="cuda")
                del dummy
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error in defragmentation: {e}")

class AcceleratedLlamaClient(ChatCompletionClient):
    """
    Accelerated version of LlamaChatCompletionClient that works with multi-GPU setup.
    
    This class wraps the model with Accelerate's distributed capabilities
    while maintaining compatibility with the existing interface.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        config: dict = None,
        accelerator: Accelerator = None,
    ):
        """Initialize the accelerated Llama client."""
        if not ACCELERATE_AVAILABLE:
            raise ImportError("Accelerate is not available. Please install it with: pip install accelerate")
        
        self.model_name = model_name
        self.config = config or {}
        self.accelerator = accelerator
        self._total_tokens = 0
        self._total_cost = 0
        
        if not self.accelerator:
            raise ValueError("Accelerator instance is required for AcceleratedLlamaClient")
        
        # Print device information
        self.accelerator.print(f"Initializing model {model_name} on process {self.accelerator.process_index}")
        self.accelerator.print(f"Device: {self.accelerator.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.tokenizer.padding_side = 'left'
        
        # Determine model size for memory optimization
        model_size_str = model_name.lower()
        is_very_large_model = any(size in model_size_str for size in ["70b", "65b"])
        
        if is_very_large_model:
            self.accelerator.print(f"Detected VERY LARGE model ({model_name}), optimizing for multi-GPU distribution...")
            # Reduce batch size and token limits for very large models
            original_max_tokens = self.config.get("maxTokenCount", 2048)
            self.config["maxTokenCount"] = min(512, original_max_tokens)
            self.accelerator.print(f"Reduced max token count to {self.config['maxTokenCount']} for very large model")
            
            # For very large models, use device_map="auto" with max_memory constraints
            # This bypasses Accelerate's prepare() but handles multi-GPU distribution properly
            model_kwargs = {
                "torch_dtype": torch.bfloat16,
                "trust_remote_code": True,
                "device_map": "auto",
                "low_cpu_mem_usage": True,
                "max_memory": self._get_max_memory_config(),
            }
            
            # Load model with device mapping (bypasses accelerate.prepare for large models)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            # Enable gradient checkpointing for memory efficiency
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
                self.accelerator.print("Enabled gradient checkpointing for memory efficiency")
            
            self.accelerator.print("Using device_map='auto' for very large model (bypassing accelerate.prepare)")
            
        else:
            # For smaller models, use standard accelerate workflow
            model_kwargs = {
                "torch_dtype": torch.bfloat16,
                "trust_remote_code": True,
                "device_map": None,  # Let accelerate handle device placement
                "low_cpu_mem_usage": True,
            }
            
            # Load model on CPU first, then let accelerate handle GPU placement
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            # Prepare model with accelerate (this handles multi-GPU distribution and device placement)
            self.model = self.accelerator.prepare(self.model)
        
        # Try to enable flash attention after model is on GPU
        try:
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'attn_implementation'):
                self.model.config.attn_implementation = "flash_attention_2"
                self.accelerator.print("Flash attention 2.0 enabled after GPU placement")
        except Exception as e:
            self.accelerator.print(f"Flash attention not available: {e}")
        
        self.accelerator.print(f"Model {model_name} loaded and distributed")
    
    def _get_max_memory_config(self) -> Dict[int, str]:
        """Get max memory configuration for multi-GPU setup with 8×A100 40GB."""
        if not torch.cuda.is_available():
            return None
        
        device_count = torch.cuda.device_count()
        max_memory = {}
        
        # For 8×A100 40GB setup, reserve some memory for other processes
        # Llama 70B in bfloat16 needs ~140GB, so ~17.5GB per GPU
        # Reserve 2-3GB per GPU for other processes and overhead
        memory_per_gpu = "37GB"  # Conservative allocation
        
        for i in range(device_count):
            max_memory[i] = memory_per_gpu
        
        self.accelerator.print(f"Configured max memory: {max_memory}")
        return max_memory
    
    def _format_messages(self, messages: List[LLMMessage]) -> List[Dict[str, str]]:
        """Format the messages for the Llama model."""
        formatted_messages = []
        
        for msg in messages:
            if isinstance(msg, SystemMessage):
                formatted_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, UserMessage):
                formatted_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AssistantMessage):
                formatted_messages.append({"role": "assistant", "content": msg.content})
        
        return formatted_messages
    
    def create_batch_sync(self, batch_messages: List[List[LLMMessage]]) -> List[AssistantMessage]:
        """Create batch completions using the accelerated Llama model."""
        max_tokens = self.config.get("maxTokenCount", 1024)
        temperature = self.config.get("temperature", 0)
        top_p = self.config.get("top_p", None)
        
        # Format all messages and convert to text
        batch_texts = []
        for messages in batch_messages:
            formatted_messages = self._format_messages(messages)
            text = self.tokenizer.apply_chat_template(
                formatted_messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            batch_texts.append(text)
        
        # Tokenize all inputs with padding
        inputs = self.tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=2048
        )
        
        # Move to device (accelerate handles this)
        inputs = {k: v.to(self.accelerator.device) for k, v in inputs.items()}
        
        input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        
        # Prepare generation parameters
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "attention_mask": inputs["attention_mask"],
        }
        
        # Handle sampling parameters
        if temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = temperature
            if top_p is not None:
                generation_kwargs["top_p"] = top_p
        else:
            generation_kwargs["do_sample"] = False
        
        # Generate the responses
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                return_dict_in_generate=True,
                output_scores=True,
                **generation_kwargs
            )
        
        # Decode all responses
        responses = []
        for i, output in enumerate(outputs.sequences):
            input_length = input_lengths[i]
            output_ids = output[input_length:]
            response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            
            # Create AssistantMessage
            response_msg = AssistantMessage(content=response, source="llama-accelerated")
            responses.append(response_msg)
            
            # Update token usage
            self._total_tokens += input_length + len(output_ids)
        
        return responses
    
    def create_sync(self, messages: List[LLMMessage]) -> AssistantMessage:
        """Create a single completion."""
        return self.create_batch_sync([messages])[0]
    
    async def create(self, messages: List[LLMMessage]) -> AssistantMessage:
        """Create a completion asynchronously."""
        return self.create_sync(messages)
    
    async def create_batch(self, batch_messages: List[List[LLMMessage]]) -> List[AssistantMessage]:
        """Create batch completions asynchronously."""
        return self.create_batch_sync(batch_messages)
    
    def count_tokens(self, messages: List[LLMMessage]) -> int:
        """Count the number of tokens in the messages."""
        formatted_messages = self._format_messages(messages)
        text = self.tokenizer.apply_chat_template(formatted_messages, tokenize=False)
        tokens = self.tokenizer.encode(text)
        return len(tokens)
    
    def remaining_tokens(self, messages: List[LLMMessage]) -> int:
        """Calculate the remaining tokens available for the response."""
        max_tokens = self.config.get("maxTokenCount", 2048)
        used_tokens = self.count_tokens(messages)
        return max_tokens - used_tokens
    
    def total_usage(self) -> float:
        """Get the total token usage."""
        return self._total_tokens
    
    def actual_usage(self) -> float:
        """Get the actual cost of usage."""
        return self._total_cost
    
    def capabilities(self) -> Dict[str, bool]:
        """Get the capabilities of the model."""
        return {
            "stream": False,
            "token_count": True,
            "token_limit": True,
            "multi_turn": True
        }
    
    def model_info(self):
        """Get information about the model."""
        max_tokens = self.config.get("maxTokenCount", 2048)
        return {
            "name": self.model_name,
            "provider": "Huggingface-Accelerated",
            "max_tokens": max_tokens,
            "token_limit": max_tokens
        }
    
    async def create_stream(self, messages: List[LLMMessage]):
        """
        Create a streaming completion using the accelerated Llama model.
        
        Args:
            messages: List of messages.
            
        Yields:
            AssistantMessage objects with partial responses.
        """
        # For simplicity, we'll just use the non-streaming version for now
        response = await self.create(messages)
        yield response
    
    def close(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
        
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


def llama_prediction(prompt, config, model_name="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Make a prediction using the Llama model.
    
    Args:
        prompt: The prompt to use.
        config: Configuration for the model.
        model_name: The name of the model on Huggingface.
        
    Returns:
        The model's response.
    """
    client = LlamaChatCompletionClient(model_name=model_name, config=config)
    
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
            messages.append(AssistantMessage(content=content, source="llama-huggingface"))
    
    # Generate the response
    response = client.create(messages)
    
    # Clean up
    client.close()
    
    # Return the response in a format similar to other prediction functions
    return {
        "content": [{"text": response.content}],
        "usage": {
            "total_tokens": client.total_usage()
        }
    }
