import pytest

from src.reasoning.output_validator import OutputValidator

VALID_RESPONSE = """

LEGAL ISSUE:

Whether an employer can terminate an employee without notice.

RELEVANT AUTHORITIES:

Case A v. Case B, as described in the supplied legal context.

LEGAL REASONING:

The supplied evidence discusses the circumstances under which notice

may be required.

CONCLUSION:

The answer depends on the applicable principles supported by the evidence.

LIMITATIONS:

The analysis is limited to the supplied legal context.

"""

def test_valid_response_passes_validation():

    result = OutputValidator.validate(VALID_RESPONSE)

    assert result["is_valid"] is True

    assert result["status"] == "valid"

    assert result["missing_sections"] == []

    assert result["empty_sections"] == []

def test_missing_section_is_detected():

    response = """

    LEGAL ISSUE:

    Whether notice is required.

    RELEVANT AUTHORITIES:

    Case A v. Case B.

    LEGAL REASONING:

    The supplied context discusses notice requirements.

    LIMITATIONS:

    The analysis is limited to the supplied context.

    """

    result = OutputValidator.validate(response)

    assert result["is_valid"] is False

    assert result["status"] == "missing_sections"

    assert result["missing_sections"] == ["CONCLUSION:"]

def test_multiple_missing_sections_are_detected():

    response = """

    LEGAL ISSUE:

    Whether notice is required.

    LEGAL REASONING:

    The supplied evidence discusses notice requirements.

    """

    result = OutputValidator.validate(response)

    assert result["is_valid"] is False

    assert result["status"] == "missing_sections"

    assert result["missing_sections"] == [

        "RELEVANT AUTHORITIES:",

        "CONCLUSION:",

        "LIMITATIONS:",

    ]

def test_empty_section_is_detected():

    response = """

        LEGAL ISSUE:

        Whether notice is required.

        RELEVANT AUTHORITIES:

        Case A v. Case B.

        LEGAL REASONING:

        The supplied context discusses notice requirements.

        CONCLUSION:

        LIMITATIONS:

        The analysis is limited to the supplied context.

        """

    result = OutputValidator.validate(response)

    assert result["is_valid"] is False

    assert result["status"] == "empty_sections"

    assert result["empty_sections"] == ["CONCLUSION:"]

def test_whitespace_only_section_is_detected():

    response = """

    LEGAL ISSUE:

    Whether notice is required.

    RELEVANT AUTHORITIES:

    Case A v. Case B.

    LEGAL REASONING:

    The supplied context discusses notice requirements.

    CONCLUSION:

        

    LIMITATIONS:

    The analysis is limited to the supplied context.

    """

    result = OutputValidator.validate(response)

    assert result["is_valid"] is False

    assert result["status"] == "empty_sections"

    assert result["empty_sections"] == ["CONCLUSION:"]

@pytest.mark.parametrize(

    "response",

    [


        "",
        "   ",

        "\n\t  \n",

    ],

)

def test_empty_or_whitespace_response_is_rejected(response):

    result = OutputValidator.validate(response)

    assert result["is_valid"] is False

    assert result["status"] == "malformed_output"

@pytest.mark.parametrize(

    "response",

    [

        None,

        123,

        [],

        {},

    ],

)

def test_non_string_input_is_rejected(response):

    result = OutputValidator.validate(response)

    assert result["is_valid"] is False

    assert result["status"] == "malformed_output"

def test_exact_insufficient_evidence_response_is_recognized():

    response = "Insufficient legal context available."

    result = OutputValidator.validate(response)

    assert result["is_valid"] is True

    assert result["status"] == "insufficient_evidence"

    assert result["missing_sections"] == []

    assert result["empty_sections"] == []

def test_insufficient_evidence_response_allows_surrounding_whitespace():

    response = "\n  Insufficient legal context available.  \n"

    result = OutputValidator.validate(response)

    assert result["is_valid"] is True

    assert result["status"] == "insufficient_evidence"

def test_similar_but_not_exact_fallback_is_not_accepted():

    response = "Insufficient legal context available. More details are needed."

    result = OutputValidator.validate(response)

    assert result["is_valid"] is False

    assert result["status"] == "missing_sections"

def test_sections_in_wrong_order_are_rejected():

    response = """

LEGAL ISSUE:

Whether notice is required.

RELEVANT AUTHORITIES:

Case A v. Case B.

CONCLUSION:

The answer depends on the evidence.

LEGAL REASONING:

The supplied evidence discusses notice requirements.

LIMITATIONS:

The analysis is limited to the supplied context.

"""

    result = OutputValidator.validate(response)

    assert result["is_valid"] is False

    assert result["status"] == "malformed_output"

def test_heading_mentioned_inside_section_does_not_create_empty_section():

    response = """

    LEGAL ISSUE:

    Whether notice is required. The question includes the phrase

    CONCLUSION: as an example of a heading.

    RELEVANT AUTHORITIES:

    Case A v. Case B.

    LEGAL REASONING:

    The supplied evidence discusses notice requirements.

    CONCLUSION:

    The answer depends on the supplied evidence.

    LIMITATIONS:

    The analysis is limited to the supplied context.

    """

    result = OutputValidator.validate(response)

    assert result["is_valid"] is True
    assert result["status"] == "valid"
