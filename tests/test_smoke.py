import pytest
from ai_zat.agent import build_workflow, GraphState

def test_imports():
    """Test that modules can be imported."""
    import ai_zat.app
    assert True

def test_workflow_build():
    """Test that the workflow builds correctly."""
    workflow = build_workflow()
    assert workflow is not None
