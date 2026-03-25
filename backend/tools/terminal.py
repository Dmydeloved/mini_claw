"""
Core Tools Implementation for Mini-OpenClaw
Wraps LangChain native tools with safety and enhancement features
"""

import os
from typing import List, Optional

from langchain_community.tools import ShellTool
from pydantic import PrivateAttr


class SafeShellTool(ShellTool):
    """
    Sandboxed shell tool with command blacklist
    """

    _root_dir: str = PrivateAttr()
    _blacklist: List[str] = PrivateAttr()

    def __init__(self, root_dir: str):
        root_dir_abs = os.path.abspath(root_dir)
        super().__init__(
            name="terminal",
            description=(
                "Execute shell commands in a sandboxed environment. "
                f"Working directory is restricted to: {root_dir_abs}. "
                "Use this to run system commands, manage files, or execute scripts."
            ),
            ask_human_input=False,
        )
        self._root_dir = root_dir_abs

        # Dangerous command blacklist
        self._blacklist = [
            "rm -rf /",
            "sudo",
            "su",
            "chmod 777",
            "mkfs",
            "dd if=",
            "> /dev/",
            ":(){ :|:& };:",  # Fork bomb
        ]

    def _run(self, command: str) -> str:
        """Execute command with safety checks"""
        # Check blacklist
        for dangerous_cmd in self._blacklist:
            if dangerous_cmd in command.lower():
                return f"Error: Command blocked for safety reasons. Dangerous pattern detected: {dangerous_cmd}"

        # Check if trying to escape root_dir
        if ".." in command or command.startswith("/"):
            return "Error: Cannot access paths outside project directory"

        try:
            # Change to root_dir before execution
            original_dir = os.getcwd()
            os.chdir(self._root_dir)

            result = super()._run(command)

            # Restore original directory
            os.chdir(original_dir)

            return result
        except Exception as e:
            return f"Error executing command: {str(e)}"


def create_terminal_tool(root_dir: Optional[str] = None) -> SafeShellTool:
    """
    Factory function to create sandboxed terminal tool

    Args:
        root_dir: Root directory for command execution (defaults to current dir)

    Returns:
        SafeShellTool instance
    """
    if root_dir is None:
        root_dir = os.getcwd()

    return SafeShellTool(root_dir=root_dir)
