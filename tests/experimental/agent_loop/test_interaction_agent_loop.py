# Copyright 2024 Bytedance Ltd. and/or its affiliates

import json
import os
import numpy as np
import pytest
import ray
from omegaconf import DictConfig


from tests.experimental.agent_loop.agent_utils import init_agent_loop_manager
from verl.protocol import DataProto

from verl.experimental.agent_loop.interaction_agent_loop import InteractionAgentLoop
from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.agent_loop import get_trajectory_info


@pytest.fixture
def scenario_config() -> DictConfig:
    """Configure parameters for multi-turn dialogue testing"""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                # Maintain original configuration overrides
                "actor_rollout_ref.rollout.n=4",
                "actor_rollout_ref.actor.use_dynamic_bsz=true",
                "actor_rollout_ref.actor.fsdp_config.param_offload=True",
                "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
                "+actor_rollout_ref.rollout.multi_turn.max_user_turns=3",
                "+actor_rollout_ref.rollout.multi_turn.termination_threshold=0.8",
                "actor_rollout_ref.rollout.agent.num_workers=4",
                # Model configuration
                "actor_rollout_ref.model.path=/file_system/common-models/Qwen/Qwen2.5-1.5B-Instruct",
                "actor_rollout_ref.rollout.name=vllm",
                "actor_rollout_ref.rollout.mode=async",
                "actor_rollout_ref.rollout.prompt_length=4096",
                "actor_rollout_ref.rollout.response_length=4096"
            ],
        )
    return config

@pytest.mark.asyncio
async def test_interaction_scenario(scenario_config):
    """End-to-end test for multi-turn conversation workflow"""
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1"
            }
        }
    )

    # Initialize manager
    agent_loop_manager = init_agent_loop_manager(scenario_config)

    # Create test data batch
    test_conversations = [
        np.array([
            {"role": "user", "content": "Recommend weekend activities in New York?"},
            {"role": "assistant", "content": "1. Visit Central Park\n2. MoMA Museum\n3. Statue of Liberty tour"},
        ], dtype=object),
    ]
    
    batch = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array(test_conversations, dtype=object),
            "agent_name": np.array(["test_interaction_agent"] * len(test_conversations)),
        }
    )
    
    # Batch expansion
    n = scenario_config.actor_rollout_ref.rollout.n
    batch = batch.repeat(n)

    # Generate sequences
    result = agent_loop_manager.generate_sequences(prompts=batch)

    # Generate index array
    index = list(range(len(result)))
    
    # Get trajectory info
    trajectory_info = await get_trajectory_info(
        step=0,
        index=index,
        validate=False
    )
    
    # Validate core metadata
    assert len(trajectory_info) == len(index)
    assert trajectory_info[0]["rollout_n"] == 0
    assert trajectory_info[-1]["rollout_n"] == 0

    ray.shutdown()


@pytest.mark.asyncio
async def test_get_trajectory_info():
    step = 10
    index = [1, 1, 3, 3]
    expected_info = [
        {"step": step, "sample_index": 1, "rollout_n": 0, "validate": False},
        {"step": step, "sample_index": 1, "rollout_n": 1, "validate": False},
        {"step": step, "sample_index": 3, "rollout_n": 0, "validate": False},
        {"step": step, "sample_index": 3, "rollout_n": 1, "validate": False},
    ]

    trajectory_info = await get_trajectory_info(step, index, validate=False)
    trajectory_info = await get_trajectory_info(step, index, validate=False)
    assert trajectory_info == expected_info