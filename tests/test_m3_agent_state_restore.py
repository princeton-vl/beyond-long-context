import os

import pytest

RUN_M3_TEST = os.environ.get("RUN_M3_AGENT_STATE_TEST") == "1"

@pytest.mark.skipif(not RUN_M3_TEST, reason="Set RUN_M3_AGENT_STATE_TEST=1 to exercise M3-Agent state restore (heavy dependencies).")
@pytest.mark.skip("M3-Agent state-restore test placeholder – enable once environment provides required dependencies.")
def test_m3_agent_state_roundtrip_placeholder():
    pass
