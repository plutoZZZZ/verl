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

    def __init__(self, **kwargs):
        """Initialize agent loop, each sample will have its own loop instance.

        Args:
            trainer_config (_DummyConfig): trainer config.
            server_manager (AsyncLLMServerManager): OpenAI compatible LLM server manager.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
        """
        super().__init__(**kwargs)
        # Initialize interaction_map for each instance
        self.interaction_map: dict[str, BaseInteraction] = self._initialize_interactions(self.config)

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> InteractionAgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
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
        # Initialize interaction
        interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]
        interaction_name = interaction_kwargs.get("name", "gsm8k")  # Default to gsm8k for backward compatibility
        if interaction_name not in self.interaction_map:
            raise ValueError(
                f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                f"{list(self.interaction_map.keys())}"
            )
        interaction = self.interaction_map[interaction_name]
        await interaction.start_interaction(request_id, **interaction_kwargs)
        # Main loop
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

            if len(response_mask) >= self.response_length:
                metrics["termination_reason"] = "max_length"
                break
            if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                metrics["termination_reason"] = "max_assistant_turns"
                break
            if self.max_user_turns and user_turns >= self.max_user_turns:
                metrics["termination_reason"] = "max_user_turns"
                break
            # Calculate turn score
            # Decode response before scoring
            loop = asyncio.get_running_loop()
            response_text = await loop.run_in_executor(
                None, 
                self.tokenizer.decode, 
                response_ids
            )
            messages.append({"role": "assistant", "content": response_text})
            # Calculate score with text response
            should_terminate_sequence, content, reward, metrics = await interaction.generate_response(
                request_id, messages, interaction_kwargs
            )
            turn_scores.append(reward)
            messages.append({"role": "user", "content": content})
            user_turns += 1
            interaction_response_ids = await self.loop.run_in_executor(
                None,
                lambda messages = content: self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True
                ),
            )
            interaction_response_ids = interaction_response_ids[len(self.system_prompt) :]
            if should_terminate_sequence:
                metrics["termination_reason"] = "interaction_terminated"
                break
            prompt_ids += interaction_response_ids
            response_mask += [0] * len(interaction_response_ids)
            user_turns += 1

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
    
    def _initialize_interactions(self, config):
        """Initialize interactions from configuration.

        Returns:
            dict[str, BaseInteraction]: A dictionary mapping interaction names to interaction instances.
        """
        if config.actor_rollout_ref.rollout.multi_turn.interaction_config_path is None:
            return {}

        interaction_config_file = config.actor_rollout_ref.rollout.multi_turn.interaction_config_path
        interaction_map = initialize_interactions_from_config(interaction_config_file)

        logger.info(f"Initialize interactions from configuration: interaction_map: {list(interaction_map.keys())}")
        return interaction_map