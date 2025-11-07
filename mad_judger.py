#!/usr/bin/env python3
"""
Multi-Agent Debate (MAD) Judger Module
Evaluates debates between multiple agents and makes final decisions.
"""

import logging
import re
import asyncio
import json
from typing import List, Dict, Any
from dataclasses import dataclass

from autogen_core import (
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    default_subscription,
    message_handler,
)
from autogen_core.models import (
    ChatCompletionClient,
    SystemMessage,
    UserMessage
)

from dataclass import (
    Answer,
    ErrorResponse
)

logger = logging.getLogger(__name__)

@dataclass
class JudgerRequest:
    """Request message for judger evaluation"""
    question: str
    debate_history: List[Dict[str, Any]]
    agent_responses_data: List[Dict[str, Any]]

@dataclass
class IntermediateJudgerRequest:
    """Request message for intermediate judger evaluation after each round"""
    question: str
    current_round: int
    agent_responses_data: List[Dict[str, Any]]

@dataclass
class JudgerResponse:
    """Response message from judger"""
    final_decision: str
    reasoning: str
    confidence: str
    agent_scores: Dict[str, float]
    raw_response: str

@dataclass
class IntermediateJudgerResponse:
    """Response message from intermediate judger evaluation"""
    should_continue: bool
    current_decision: str
    reasoning: str
    confidence: str
    round_evaluated: int
    raw_response: str

@default_subscription
class MADJudger(RoutedAgent):
    """Judger agent that evaluates multi-agent debates and makes final decisions"""
    
    def __init__(self, model_client: ChatCompletionClient, prompt_config: Dict[str, Any], 
                 result_queue=None):
        super().__init__("MAD Judger")
        self._model_client = model_client
        self._prompt_config = prompt_config
        self._result_queue = result_queue
        self._system_prompt = self._create_judger_system_prompt()
        
    def _create_judger_system_prompt(self) -> str:
        """Create system prompt for the judger"""
        return "You are a moderator evaluating a debate between two agents. Analyze their arguments and determine the best answer."

    async def _call_model_with_retry(self, messages: List) -> Any:
        """Call model with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return await asyncio.wait_for(
                    self._model_client.create(messages),
                    timeout=300
                )
            except asyncio.TimeoutError:
                if attempt == max_retries - 1:
                    raise ValueError("Model response timed out after multiple attempts")
                await asyncio.sleep(1)
            except Exception as e:
                raise ValueError(f"Model error: {str(e)}")

    @message_handler
    async def handle_judger_request(self, message: JudgerRequest, ctx: MessageContext) -> None:
        """Handle judger evaluation request"""
        try:
            debate_summary = self._format_debate_history(message.question, message.agent_responses_data)
            evaluation_prompt = self._create_evaluation_prompt(debate_summary)
            
            messages = [
                SystemMessage(content=self._system_prompt),
                UserMessage(content=evaluation_prompt, source="user")
            ]
            
            model_result = await self._call_model_with_retry(messages)
            judgment = self._parse_judgment(model_result.content)
            
            judger_response = JudgerResponse(
                final_decision=judgment["decision"],
                reasoning=judgment["reasoning"],
                raw_response=model_result.content
            )
            
            await self.publish_message(judger_response, topic_id=DefaultTopicId())
            await self.publish_message(Answer(content=judgment["decision"]), topic_id=DefaultTopicId())
            
            if self._result_queue:
                result = {
                    'numerical_answer': judgment["decision"],
                    'full_response': model_result.content,
                    'processing_time': 0,
                    'individual_answers': [resp_data.get('answer') for resp_data in message.agent_responses_data],
                    'judger_evaluation': {
                        'decision': judgment["decision"],
                        'reasoning': judgment["reasoning"],
                    }
                }
                await asyncio.shield(self._result_queue.put(result))
            
        except Exception as e:
            await self._handle_error(f"Error in judger evaluation: {str(e)}")

    @message_handler
    async def handle_intermediate_judger_request(self, message: IntermediateJudgerRequest, ctx: MessageContext) -> None:
        """Handle intermediate judger evaluation request after each round"""
        try:
            debate_summary = self._format_debate_history(message.question, message.agent_responses_data)
            evaluation_prompt = self._create_intermediate_evaluation_prompt(debate_summary, message.current_round)
            
            messages = [
                SystemMessage(content=self._system_prompt),
                UserMessage(content=evaluation_prompt, source="user")
            ]
            
            model_result = await self._call_model_with_retry(messages)
            judgment = self._parse_intermediate_judgment(model_result.content)
            
            intermediate_response = IntermediateJudgerResponse(
                should_continue=judgment["should_continue"],
                current_decision=judgment["current_decision"],
                reasoning=judgment["reasoning"],
                round_evaluated=message.current_round,
                raw_response=model_result.content
            )
            
            await self.publish_message(intermediate_response, topic_id=DefaultTopicId())
            
        except Exception as e:
            await self._handle_error(f"Error in intermediate judger evaluation: {str(e)}")
    
    def _format_debate_history(self, question: str, agent_responses_data: List[Dict[str, Any]]) -> str:
        """Format debate history for judger evaluation with side names"""
        # Determine which agent is affirmative and which is negative based on sender_id
        affirmative_data = None
        negative_data = None
        
        for resp_data in agent_responses_data:
            sender_id = resp_data.get('sender_id', '')
            if 'AFFIRMATIVE' in str(sender_id).upper() or 'affirmative' in str(sender_id).lower():
                affirmative_data = resp_data
            elif 'NEGATIVE' in str(sender_id).upper() or 'negative' in str(sender_id).lower():
                negative_data = resp_data
            else:
                # Fallback: assign based on order (first is affirmative, second is negative)
                if affirmative_data is None:
                    affirmative_data = resp_data
                elif negative_data is None:
                    negative_data = resp_data
        
        # Get the maximum number of rounds
        max_rounds = 0
        for resp_data in agent_responses_data:
            answers_by_round = resp_data.get('answers_by_round', {})
            if answers_by_round:
                max_rounds = max(max_rounds, len(answers_by_round))
        
        # If no rounds data, use 1 round with final answers
        if max_rounds == 0:
            max_rounds = 1
        
        # Format using the moderator_prompt template for the final round
        final_round = max_rounds
        
        # Get affirmative and negative answers for the final round
        aff_answer = "No answer provided"
        neg_answer = "No answer provided"
        
        if affirmative_data:
            answers_by_round = affirmative_data.get('answers_by_round', {})
            if answers_by_round:
                # Get the last round's answer
                aff_answer = list(answers_by_round.values())[-1] if answers_by_round else affirmative_data.get('answer', 'No answer provided')
            else:
                aff_answer = affirmative_data.get('answer', 'No answer provided')
        
        if negative_data:
            answers_by_round = negative_data.get('answers_by_round', {})
            if answers_by_round:
                # Get the last round's answer
                neg_answer = list(answers_by_round.values())[-1] if answers_by_round else negative_data.get('answer', 'No answer provided')
            else:
                neg_answer = negative_data.get('answer', 'No answer provided')
        
        # Use the moderator_prompt template
        moderator_prompt = f"Now the {final_round} round of debate for both sides has ended.\n\nAffirmative side arguing:\n{aff_answer}\n\nNegative side arguing: {neg_answer}\n\n "
        
        return moderator_prompt
    
    def _create_evaluation_prompt(self, debate_summary: str) -> str:
        """Create evaluation prompt for the judger"""
        answer_format = "Now, what answer candidates do we have? \n\n please output your answer in json format, with the format as follows: {{\"Supported Side\": \"Affirmative or Negative\", \"Reason\": \"\", \"debate_answer\": \"\"}}"
        return f"{debate_summary}\n{answer_format}"
    
    
    def _create_intermediate_evaluation_prompt(self, debate_summary: str, current_round: int) -> str:
        """Create evaluation prompt for intermediate judger evaluation"""
        answer_format = "You, as the moderator, will evaluate both sides' answers and determine if there is a clear preference for an answer candidate. If so, please summarize your reasons for supporting affirmative/negative side and give the final answer that you think is correct, and the debate will conclude. If not, the debate will continue to the next round. \n\n Now please output your answer in json format, with the format as follows: {{\"Whether there is a preference\": \"Yes or No\", \"Supported Side\": \"Affirmative or Negative\", \"Reason\": \"\", \"debate_answer\": \"\"}}. Please strictly output in JSON format, do not output irrelevant content."
        return f"{debate_summary}\n{answer_format}"
    
    def _parse_intermediate_judgment(self, response: str) -> Dict[str, Any]:
        """Parse structured judgment from intermediate judger response (JSON format)"""
        judgment = {
            "should_continue": True,  # Default to continue
            "current_decision": "",
            "reasoning": "",
            "raw_response": response if isinstance(response, str) else str(response)
        }
        
        try:
            # Try to extract JSON from the response using a more robust pattern
            # Look for JSON structure that may contain nested braces in values
            json_match = re.search(r'\{(?:[^{}]|"[^"]*")*"debate_answer"(?:[^{}]|"[^"]*")*\}', response, re.DOTALL)
            if not json_match:
                # Try a simpler approach - find the outermost braces
                start_brace = response.find('{')
                if start_brace != -1:
                    brace_count = 0
                    end_pos = start_brace
                    for i, char in enumerate(response[start_brace:], start_brace):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_pos = i + 1
                                break
                    if brace_count == 0:
                        json_match = type('Match', (), {'group': lambda self, x: response[start_brace:end_pos]})()
            
            if json_match:
                json_str = json_match.group(0)
                json_data = json.loads(json_str)
                
                # Extract decision from debate_answer field
                debate_answer = json_data.get("debate_answer", "")
                # Remove double curly braces if present
                if debate_answer.startswith("{{") and debate_answer.endswith("}}"):
                    judgment["current_decision"] = debate_answer[2:-2]
                elif debate_answer.startswith("{") and debate_answer.endswith("}"):
                    judgment["current_decision"] = debate_answer[1:-1]
                else:
                    judgment["current_decision"] = debate_answer
                
                # Extract reasoning from Reason field
                judgment["reasoning"] = json_data.get("Reason", "")
                
                # Determine if debate should continue based on preference
                preference = json_data.get("Whether there is a preference", "No")
                judgment["should_continue"] = preference.lower() == "no"
                
            else:
                # Fallback to pattern extraction
                pattern = self._prompt_config.get("answer_pattern", r"\{([^}]+)\}")
                match = re.search(pattern, response)
                if match:
                    judgment["current_decision"] = match.group(1)
                    judgment["reasoning"] = "Pattern extraction used"
                    
        except (json.JSONDecodeError, Exception) as e:
            # Final fallback
            pattern = self._prompt_config.get("answer_pattern", r"\{([^}]+)\}")
            match = re.search(pattern, response)
            if match:
                judgment["current_decision"] = match.group(1)
            judgment["reasoning"] = "Parsing failed, used pattern fallback"
        
        return judgment
    
    def _parse_judgment(self, response: str) -> Dict[str, Any]:
        """Parse structured judgment from judger response (JSON format)"""
        judgment = {
            "decision": "",
            "reasoning": "",
            "raw_response": response
        }
        
        try:
            # Try to extract JSON from the response using a more robust pattern
            # Look for JSON structure that may contain nested braces in values
            json_match = re.search(r'\{(?:[^{}]|"[^"]*")*"debate_answer"(?:[^{}]|"[^"]*")*\}', response, re.DOTALL)
            if not json_match:
                # Try a simpler approach - find the outermost braces
                start_brace = response.find('{')
                if start_brace != -1:
                    brace_count = 0
                    end_pos = start_brace
                    for i, char in enumerate(response[start_brace:], start_brace):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_pos = i + 1
                                break
                    if brace_count == 0:
                        json_match = type('Match', (), {'group': lambda self, x: response[start_brace:end_pos]})()
            
            if json_match:
                json_str = json_match.group(0)
                json_data = json.loads(json_str)
                
                # Extract decision from debate_answer field
                debate_answer = json_data.get("debate_answer", "")
                # Remove double curly braces if present
                if debate_answer.startswith("{{") and debate_answer.endswith("}}"):
                    judgment["decision"] = debate_answer[2:-2]
                elif debate_answer.startswith("{") and debate_answer.endswith("}"):
                    judgment["decision"] = debate_answer[1:-1]
                else:
                    judgment["decision"] = debate_answer
                
                # Extract reasoning from Reason field
                judgment["reasoning"] = json_data.get("Reason", "")
                
            else:
                # Fallback to pattern extraction
                pattern = self._prompt_config.get("answer_pattern", r"\{([^}]+)\}")
                match = re.search(pattern, response)
                if match:
                    judgment["decision"] = match.group(1)
                    judgment["reasoning"] = "Pattern extraction used"
                    
        except (json.JSONDecodeError, Exception) as e:
            # Final fallback
            pattern = self._prompt_config.get("answer_pattern", r"\{([^}]+)\}")
            match = re.search(pattern, response)
            if match:
                judgment["decision"] = match.group(1)
            judgment["reasoning"] = "Parsing failed, used pattern fallback"
        
        return judgment
    
    async def _handle_error(self, error_msg: str) -> None:
        """Handle errors during judger evaluation"""
        await self.publish_message(
            ErrorResponse(content=error_msg),
            topic_id=DefaultTopicId()
        )
