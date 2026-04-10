"""
File Writer Tool for Mini-OpenClaw
Sandboxed file writing for skills and other local text files.
"""

import os
import re
from typing import Optional

import yaml
from langchain_community.tools.file_management import WriteFileTool


class SafeWriteFileTool(WriteFileTool):
    """
    Sandboxed file writer restricted to project directory.
    """

    def __init__(self, root_dir: str):
        root_dir_abs = os.path.abspath(root_dir)
        super().__init__(
            root_dir=root_dir_abs,
            name="write_file",
            description=(
                "Write or overwrite a text file inside the project directory. "
                f"File paths are relative to: {root_dir_abs}. "
                "Use this to create or update local files such as "
                "`skills/<skill_name>/SKILL.md`, config files, or generated text artifacts."
            ),
        )

    def _is_skill_file(self, file_path: str) -> bool:
        normalized = file_path.replace("\\", "/").lstrip("./")
        return normalized.startswith("skills/") and normalized.endswith("/SKILL.md")

    def _validate_skill_content(self, text: str) -> Optional[str]:
        frontmatter_pattern = r"^---\s*\n(.*?)\n---\s*\n(.*)$"
        match = re.match(frontmatter_pattern, text, re.DOTALL)

        if not match:
            return (
                "Error: Skill files must include YAML frontmatter wrapped in `---` markers, "
                "followed by the Markdown body."
            )

        try:
            metadata = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError as exc:
            return f"Error: Invalid YAML frontmatter in skill file: {exc}"

        missing_fields = [field for field in ("name", "description") if not metadata.get(field)]
        if missing_fields:
            return (
                "Error: Skill files must define the following frontmatter fields: "
                + ", ".join(missing_fields)
                + "."
            )

        if not match.group(2).strip():
            return "Error: Skill files must include Markdown instructions after the frontmatter."

        return None

    def _run(self, file_path: str, text: str, append: bool = False) -> str:
        """Write a file with path checks and skill validation."""
        try:
            root_dir = os.path.abspath(self.root_dir or ".")
            abs_path = os.path.abspath(os.path.join(root_dir, file_path))

            if not abs_path.startswith(root_dir):
                return f"Error: Access denied. File must be within {root_dir}"

            if self._is_skill_file(file_path):
                if append:
                    return "Error: Appending to SKILL.md is not allowed. Rewrite the full skill file instead."

                validation_error = self._validate_skill_content(text)
                if validation_error:
                    return validation_error

            return super()._run(file_path=file_path, text=text, append=append)
        except Exception as e:
            return f"Error writing file: {str(e)}"


def create_write_file_tool(root_dir: Optional[str] = None) -> SafeWriteFileTool:
    """
    Factory function to create file writer tool

    Args:
        root_dir: Root directory for file writes (defaults to current dir)

    Returns:
        SafeWriteFileTool instance
    """
    if root_dir is None:
        root_dir = os.getcwd()

    return SafeWriteFileTool(root_dir=root_dir)
