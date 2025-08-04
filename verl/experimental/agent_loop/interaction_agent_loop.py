import asyncio
import logging
import os
from typing import List, Dict, Any
from uuid import uuid4
from pydantic import BaseModel

from .agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class InteractionAgentLoopOutput(AgentLoopOutput):
    turn_scores: List[float] = []
    interaction_final_scores: Dict[str, float] = {}

@register("interaction_agent")
class InteractionAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, **kwargs):
        if cls._class_initialized:
            return
            
        cls._class_initialized = True
        print("Performing class-level InteractionAgentLoop initialization")

        # Initialize system prompt
        cls.tokenizer = tokenizer
        cls.system_prompt = tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True
        )
        
        # Basic parameters
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.termination_threshold = config.actor_rollout_ref.rollout.get("termination_threshold", 0.8)

    def __init__(self, **kwargs):
        """Initialize agent loop, each sample will have its own loop instance.

        Args:
            trainer_config (_DummyConfig): trainer config.
            server_manager (AsyncLLMServerManager): OpenAI compatible LLM server manager.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
        """
        super().__init__(**kwargs)
        # Initialize interaction_map for each instance
        self.interaction_map: dict[str, BaseInteraction] = initialize_interactions_from_config(interaction_config_file=self.config.actor_rollout_ref.rollout.multi_turn.interaction_config_path)
        # Initialize interaction_instances to store instance IDs
        self.interaction_instances: dict[str, str] = {}
        for name, interaction in self.interaction_map.items():
            instance_id = interaction.start_interaction()
            self.interaction_instances[name] = instance_id

    @rollout_trace_op
    async def run(self, messages: List[Dict[str, Any]], sampling_params: Dict[str, Any]) -> InteractionAgentLoopOutput:
        metrics = {}
        request_id = uuid4().hex
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True
            ),
        )
        response_mask = []
        turn_scores = []
        user_turns, assistant_turns = 0, 0

        while True:
            # Generate assistant response
            with simple_timer("generate_sequences", metrics):
                response_ids = await self.server_manager.generate(
                    request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params
                )
            
            # Update conversation state
            prompt_ids += response_ids
            response_mask += [1] * len(response_ids)
            assistant_turns += 1

            # Calculate turn score
            # Decode response before scoring
            loop = asyncio.get_running_loop()
            response_text = await loop.run_in_executor(
                None, 
                self.tokenizer.decode, 
                response_ids
            )
            
            # Calculate score with text response
            turn_score = await self.calculate_turn_score(response_text)
            turn_scores.append(turn_score)

            # Termination conditions (keep same check order as tool_agent)
            if turn_score >= self.termination_threshold:
                metrics["termination_reason"] = "threshold_reached"
                break
            if len(response_mask) >= self.response_length:
                metrics["termination_reason"] = "max_length"
                break
            if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                metrics["termination_reason"] = "max_assistant_turns"
                break
            if self.max_user_turns and user_turns >= self.max_user_turns:
                metrics["termination_reason"] = "max_user_turns"
                break

        # Prepare output (aligned with tool_agent's output structure)
        response_ids = prompt_ids[-len(response_mask):]
        prompt_ids = prompt_ids[: len(prompt_ids) - len(response_mask)]

        return InteractionAgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[:self.response_length],
            response_mask=response_mask[:self.response_length],
            num_turns=user_turns + assistant_turns,
            metrics=metrics,
            turn_scores=turn_scores,
        )

    async def calculate_turn_score(self, response_text: str) -> float:
        """Calculate turn score using interactions"""
        total_score = 0.0
        count = 0
        for name, interaction in self.interaction_map.items():
            instance_id = self.interaction_instances[name]
            # Create a simple message structure for the interaction
            messages = [{"role": "assistant", "content": response_text}]
            # Call generate_response to get the turn score
            should_terminate_sequence, content, reward, metrics = await interaction.generate_response(instance_id, messages)
            total_score += reward
            count += 1
        return total_score / count if count > 0 else 0.0