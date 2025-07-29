# test_interaction_agent_loop.py
import pytest
from verl.experimental.agent_loop.interaction_agent_loop import InteractionAgentLoop
from unittest.mock import Mock, AsyncMock

class TestInteractionAgentLoop(InteractionAgentLoop):
    """Test subclass that implements abstract methods"""
    def __init__(self):
        # Mock necessary parameters
        mock_config = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.decode.return_value = ""  # Default return empty string
        
        # Initialize base class
        super().__init__(
            trainer_config=mock_config,
            server_manager=AsyncMock(),
            tokenizer=mock_tokenizer
        )
        
    async def calculate_turn_score(self, response_text: str) -> float:
        """Simple implementation: score based on text length"""
        return len(response_text) * 0.1

@pytest.mark.asyncio
async def test_calculate_turn_score_normal():
    """Test normal input scenarios"""
    test_agent = TestInteractionAgentLoop()
    
    # Test non-empty string
    score = await test_agent.calculate_turn_score("Hello world")
    assert score == 1.1  # 11 chars * 0.1

    # Test special characters
    score = await test_agent.calculate_turn_score("测试@#¥%")
    assert score == pytest.approx(0.6)  # 6 actual chars * 0.1 = 0.6
    
    # Test long text
    long_text = "a" * 1000
    score = await test_agent.calculate_turn_score(long_text)
    assert score == 100.0  # 1000 * 0.1

@pytest.mark.asyncio
async def test_calculate_turn_score_edge_cases():
    """Test edge cases"""
    test_agent = TestInteractionAgentLoop()
    
    # Test empty string
    score = await test_agent.calculate_turn_score("")
    assert score == 0.0
    
    # Test whitespace string
    score = await test_agent.calculate_turn_score("   ")
    assert score == pytest.approx(0.3)  # 3 spaces * 0.1

@pytest.mark.asyncio
async def test_calculate_turn_score_error_handling():
    """Test error handling"""
    test_agent = TestInteractionAgentLoop()
    
    # Test None input
    with pytest.raises(TypeError):
        await test_agent.calculate_turn_score(None)
    
    # Test non-string input
    with pytest.raises(TypeError):
        await test_agent.calculate_turn_score(123)

def test_abstract_method_implementation():
    """Test abstract method implementation requirements"""
    with pytest.raises(TypeError):
        agent = InteractionAgentLoop()  # Abstract class shouldn't be instantiated
    
    # Verify subclass must implement abstract methods
    class InvalidSubclass(InteractionAgentLoop):
        pass
    
    with pytest.raises(TypeError):
        InvalidSubclass()