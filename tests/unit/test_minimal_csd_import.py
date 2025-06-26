
import pytest

def test_minimal_csd_import():
    from enhanced_rag_csd.core.csd_emulator import EnhancedCSDSimulator
    assert EnhancedCSDSimulator is not None
