import re


class OutputValidator:
    """
    Validate the structure of generated LegalRAG responses.

    This validator checks required section headings and section content.
    It does not verify legal correctness or citation authenticity.
    """

    REQUIRED_SECTIONS = (
        "LEGAL ISSUE:",
        "RELEVANT AUTHORITIES:",
        "LEGAL REASONING:",
        "CONCLUSION:",
        "LIMITATIONS:",
    )

    INSUFFICIENT_EVIDENCE_RESPONSE = (
        "Insufficient legal context available."
    )

    @staticmethod
    def validate(response: str) -> dict:
        """
        Validate a generated legal response.

        Returns a dictionary containing:
        - is_valid: whether the response satisfies structural validation
        - status: the validation outcome
        - missing_sections: required headings that are missing
        - empty_sections: headings whose sections contain no content
        """

        if not isinstance(response, str):
            return {
                "is_valid": False,
                "status": "malformed_output",
                "missing_sections": list(
                    OutputValidator.REQUIRED_SECTIONS
                ),
                "empty_sections": [],
            }

        response = response.strip()

        if not response:
            return {
                "is_valid": False,
                "status": "malformed_output",
                "missing_sections": list(
                    OutputValidator.REQUIRED_SECTIONS
                ),
                "empty_sections": [],
            }

        if response == OutputValidator.INSUFFICIENT_EVIDENCE_RESPONSE:
            return {
                "is_valid": True,
                "status": "insufficient_evidence",
                "missing_sections": [],
                "empty_sections": [],
            }

        # Match headings only when they appear on their own lines.
        heading_pattern = re.compile(
            r"^[ \t]*(LEGAL ISSUE:|RELEVANT AUTHORITIES:|"
            r"LEGAL REASONING:|CONCLUSION:|LIMITATIONS:)[ \t]*$",
            re.MULTILINE,
        )

        heading_matches = list(heading_pattern.finditer(response))

        # Store each actual heading and its position in the response.
        positions = {
            match.group(1): match.start()
            for match in heading_matches
        }

        missing_sections = [
            section
            for section in OutputValidator.REQUIRED_SECTIONS
            if section not in positions
        ]

        if missing_sections:
            return {
                "is_valid": False,
                "status": "missing_sections",
                "missing_sections": missing_sections,
                "empty_sections": [],
            }

        # Ensure that headings appear in the required order.
        heading_positions = [
            positions[section]
            for section in OutputValidator.REQUIRED_SECTIONS
        ]

        if heading_positions != sorted(heading_positions):
            return {
                "is_valid": False,
                "status": "malformed_output",
                "missing_sections": [],
                "empty_sections": [],
            }

        # Check the content between consecutive headings.
        # Check the content between consecutive heading lines.

        empty_sections = []

        for index, match in enumerate(heading_matches):

            section = match.group(1)

            # The section content starts after the complete heading line.

            content_start = match.end()

            # Find where the next heading line begins.

            if index + 1 < len(heading_matches):

                content_end = heading_matches[index + 1].start()

            else:

                content_end = len(response)

            section_content = response[

                content_start:content_end

            ].strip()

            if not section_content:

                empty_sections.append(section)

        if empty_sections:

            return {

                "is_valid": False,

                "status": "empty_sections",

                "missing_sections": [],

                "empty_sections": empty_sections,

            }

        return {

            "is_valid": True,

            "status": "valid",

            "missing_sections": [],

            "empty_sections": [],

        }