# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import unittest
from unittest.mock import Mock, AsyncMock, patch, MagicMock

import ray
import torch
from omegaconf import DictConfig

from verl.experimental.agent_loop.interaction_agent_loop import InteractionAgentLoop
from verl.experimental.agent_loop.agent_loop import _DummyConfig


class TestInteractionAgentLoop(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a new event loop for each test
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        # Mock server_manager
        self.server_manager = Mock()
        self.server_manager.generate = AsyncMock()
        self.server_manager.generate.return_value = [5, 6, 7, 0, 0]

        # Mock tokenizer
        self.tokenizer = Mock()
        self.tokenizer.apply_chat_template = Mock(return_value=[1, 2, 3])
        self.tokenizer.encode = Mock(return_value=[8, 9, 10])
        self.tokenizer.decode = Mock(return_value="mocked decoded response")

        # Mock trainer_config with a DictConfig wrapped in _DummyConfig
        dict_config = DictConfig({
            "actor_rollout_ref": {
                "rollout": {
                    "multi_turn": {
                        "max_user_turns": 10,
                        "max_assistant_turns": 10,
                        "interaction_config_path": None
                    },
                    "prompt_length": 256,
                    "response_length": 256
                }
            }
        })
        self.trainer_config = _DummyConfig(config=dict_config)

        # Initialize InteractionAgentLoop class attributes
        InteractionAgentLoop.init_class(dict_config, self.tokenizer)

        # Initialize InteractionAgentLoop with mocked dependencies
        # We need to run the __init__ method in the event loop
        self.agent_loop = self.loop.run_until_complete(self._init_agent_loop())

        # Mock interaction_map with a gsm8k interaction
        mock_interaction = Mock()
        mock_interaction.start_interaction = AsyncMock(return_value=None)
        mock_interaction.generate_response = AsyncMock(return_value=(True, "Final Answer: 4", 1.0, {}))
        self.agent_loop.interaction_map = {"gsm8k": mock_interaction}

    async def _init_agent_loop(self):
        """Helper method to initialize InteractionAgentLoop in async context."""
        return InteractionAgentLoop(
            trainer_config=self.trainer_config,
            server_manager=self.server_manager,
            tokenizer=self.tokenizer
        )

    def tearDown(self):
        """Clean up after each test method."""
        # Close the event loop
        self.loop.close()

    def _run_test(self, raw_prompt, interaction_kwargs=None):
        """Helper method to run the agent loop with given raw_prompt."""
        if interaction_kwargs is None:
            interaction_kwargs = {"instance_id": "test_instance", "ground_truth": "4"}
        
        # Mock batch data
        batch = {
            "raw_prompt": raw_prompt,
            "extra_info": {
                "interaction_kwargs": interaction_kwargs
            }
        }

        # Run the agent loop
        result = self.loop.run_until_complete(self.agent_loop.run(sampling_params={}, **batch))
        return result

    def test_run_method(self):
        """Test the run method of InteractionAgentLoop."""
        raw_prompt = [{"role": "user", "content": "What is 2 + 2?"}]
        result = self._run_test(raw_prompt)

        # Assertions
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'prompt_ids'))
        self.assertTrue(hasattr(result, 'response_ids'))
        self.assertTrue(hasattr(result, 'response_mask'))
        self.assertTrue(hasattr(result, 'num_turns'))
        self.assertTrue(hasattr(result, 'metrics'))

        # Verify server_manager.generate was called
        self.server_manager.generate.assert_called()

        # Verify tokenizer methods were called
        self.tokenizer.apply_chat_template.assert_called()
        self.tokenizer.decode.assert_called()

    def test_full_interaction_workflow_correct(self):
        """Test full interaction workflow with a correct answer."""
        raw_prompt = [{"role": "user", "content": "What is 2 + 2?"}]
        result = self._run_test(raw_prompt)

        # Assertions
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'prompt_ids'))
        self.assertTrue(hasattr(result, 'response_ids'))
        self.assertTrue(hasattr(result, 'response_mask'))
        self.assertTrue(hasattr(result, 'num_turns'))
        self.assertTrue(hasattr(result, 'metrics'))

        # Verify interaction methods were called
        interaction = self.agent_loop.interaction_map["gsm8k"]
        interaction.start_interaction.assert_called_once()
        interaction.generate_response.assert_called_once()

    def test_full_interaction_workflow_incorrect(self):
        """Test full interaction workflow with an incorrect answer."""
        # Modify mock to simulate incorrect answer
        self.agent_loop.interaction_map["gsm8k"].generate_response.return_value = (False, "Incorrect Answer", 0.0, {})

        raw_prompt = [{"role": "user", "content": "What is 3 + 3?"}]
        result = self._run_test(raw_prompt)

        # Assertions
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'prompt_ids'))
        self.assertTrue(hasattr(result, 'response_ids'))
        self.assertTrue(hasattr(result, 'response_mask'))
        self.assertTrue(hasattr(result, 'num_turns'))
        self.assertTrue(hasattr(result, 'metrics'))

        # Verify interaction methods were called
        interaction = self.agent_loop.interaction_map["gsm8k"]
        interaction.start_interaction.assert_called_once()
        # Note: generate_response may be called multiple times depending on the loop logic
        interaction.generate_response.assert_called()

    def test_interaction_not_found(self):
        """Test behavior when interaction is not found."""
        with self.assertRaises(ValueError) as context:
            self._run_test(
                [{"role": "user", "content": "What is 2 + 2?"}],
                {"name": "nonexistent", "instance_id": "test_instance", "ground_truth": "4"}
            )

        self.assertIn("Interaction 'nonexistent' not found", str(context.exception))


if __name__ == '__main__':
    unittest.main()