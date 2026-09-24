from core.configuration_proposal import (
    build_configuration_proposal,
)


def test_configuration_proposal_detects_changes():
    proposal = build_configuration_proposal(
        "gemma4:e4b",
        "FROM gemma4:e4b\n",
        {
            "temperature": 1.0,
            "top_k": 64,
        },
        {
            "temperature": 0.7,
            "top_k": 40,
        },
    )

    assert proposal.model == "gemma4:e4b"
    assert proposal.has_changes
    assert len(proposal.changes) == 2
    assert "temperature" in proposal.changes[0]


def test_configuration_proposal_without_changes():
    proposal = build_configuration_proposal(
        "gemma4:e4b",
        "FROM gemma4:e4b\n",
        {
            "temperature": 1.0,
        },
        {
            "temperature": 1.0,
        },
    )

    assert not proposal.has_changes
    assert proposal.changes == []
    assert "FROM gemma4:e4b" in proposal.modelfile
