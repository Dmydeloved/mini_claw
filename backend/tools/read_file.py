"""
File Reader Tool for Mini-OpenClaw
Sandboxed file reading for Agent Skills and local files
"""

import os
from typing import Optional

from langchain_community.tools.file_management import ReadFileTool

class SafeReadFileTool(ReadFileTool):
    """
    Sandboxed file reader restricted to project directory
    """

    def __init__(self, root_dir: str):
        root_dir_abs = os.path.abspath(root_dir)
        super().__init__(
            root_dir=root_dir_abs,
            name="read_file",
            description=(
                "Read the contents of a file from the local filesystem. "
                f"File paths are relative to: {root_dir_abs}. "
                "Use this to read SKILL.md files, configuration files, or other text files."
            ),
        )

    def _run(self, file_path: str) -> str:
        """Read file with safety checks"""
        try:
            # Resolve absolute path
            root_dir = os.path.abspath(self.root_dir or ".")
            abs_path = os.path.abspath(os.path.join(root_dir, file_path))

            # Ensure path is within root_dir
            if not abs_path.startswith(root_dir):
                return f"Error: Access denied. File must be within {root_dir}"

            # Check if file exists
            if not os.path.exists(abs_path):
                return f"Error: File not found: {file_path}"

            # Check if it's a file (not directory)
            if not os.path.isfile(abs_path):
                return f"Error: Path is not a file: {file_path}"

            # Read file content
            with open(abs_path, "r", encoding="utf-8") as f:
                content = f.read()

            return content

        except Exception as e:
            return f"Error reading file: {str(e)}"


def create_read_file_tool(root_dir: Optional[str] = None) -> SafeReadFileTool:
    """
    Factory function to create file reader tool

    Args:
        root_dir: Root directory for file access (defaults to current dir)

    Returns:
        SafeReadFileTool instance
    """
    if root_dir is None:
        root_dir = os.getcwd()

    return SafeReadFileTool(root_dir=root_dir)
