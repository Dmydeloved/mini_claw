"""
Python REPL Tool for Mini-OpenClaw
Provides code execution capabilities to the Agent
"""

from langchain_experimental.tools import PythonREPLTool

class SafePythonREPLTool(PythonREPLTool):
    """
    Python REPL with enhanced description
    """

    def __init__(self):
        super().__init__(
            name="python_repl",
            description=(
                "Execute Python code in an isolated REPL environment. "
                "Use this for calculations, data processing, parsing, or quick scripts."
            ),
        )


def create_python_repl_tool() -> SafePythonREPLTool:
    """
    Factory function to create Python REPL tool

    Returns:
        SafePythonREPLTool instance
    """
    return SafePythonREPLTool()
