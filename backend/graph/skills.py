"""
Agent Skills System for Mini-OpenClaw
Handles skill loading, parsing, and snapshot generation.
"""

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass
class Skill:
    """Represents an Agent Skill"""

    name: str
    description: str
    trigger: str
    enabled: bool
    content: str  # Full markdown content including instructions
    file_path: str
    relative_path: str


class SkillManager:
    """
    Manages Agent Skills lifecycle:
    - Scanning skills directory
    - Parsing SKILL.md files
    - Matching triggers
    - Injecting skill content into prompts
    """

    def __init__(self, skills_dir: str, root_dir: str):
        self.skills_dir = os.path.abspath(skills_dir)
        self.root_dir = os.path.abspath(root_dir)
        self.skills: List[Skill] = []
        self._load_skills(verbose=True)

    def _load_skills(self, verbose: bool = False):
        """Scan skills directory and load all SKILL.md files"""
        if not os.path.exists(self.skills_dir):
            os.makedirs(self.skills_dir)
            if verbose:
                print(f"Created skills directory: {self.skills_dir}")
            return

        for skill_folder in sorted(os.listdir(self.skills_dir)):
            skill_path = os.path.join(self.skills_dir, skill_folder)

            # Skip if not a directory
            if not os.path.isdir(skill_path):
                continue

            # Look for SKILL.md
            skill_file = os.path.join(skill_path, "SKILL.md")
            if not os.path.exists(skill_file):
                if verbose:
                    print(f"Warning: No SKILL.md found in {skill_folder}")
                continue

            try:
                skill = self._parse_skill_file(skill_file)
                if skill:
                    self.skills.append(skill)
                    if verbose:
                        print(f"Loaded skill: {skill.name}")
            except Exception as e:
                if verbose:
                    print(f"Error loading skill from {skill_file}: {e}")

    def _split_skill_content(self, content: str) -> Optional[Tuple[Dict[str, Any], str]]:
        """Split SKILL.md content into YAML metadata and Markdown body."""
        frontmatter_pattern = r"^---\s*\n(.*?)\n---\s*\n(.*)$"
        match = re.match(frontmatter_pattern, content, re.DOTALL)

        if not match:
            return None

        frontmatter_text = match.group(1)
        body = match.group(2)

        try:
            metadata = yaml.safe_load(frontmatter_text) or {}
        except yaml.YAMLError:
            return None

        if not isinstance(metadata, dict):
            return None

        return metadata, body

    def _validate_skill_metadata(
        self, metadata: Dict[str, Any], source: str = "<memory>"
    ) -> Optional[str]:
        """Validate required skill metadata fields."""
        required_fields = ["name", "description"]
        for field in required_fields:
            if field not in metadata or metadata[field] in (None, ""):
                return f"Missing required field '{field}' in {source}"
        return None

    def _parse_skill_file(self, file_path: str) -> Optional[Skill]:
        """Parse SKILL.md file with YAML frontmatter"""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        split_result = self._split_skill_content(content)
        if not split_result:
            print(f"Warning: No valid frontmatter in {file_path}")
            return None

        metadata, body = split_result
        validation_error = self._validate_skill_metadata(metadata, source=file_path)
        if validation_error:
            print(validation_error)
            return None

        return Skill(
            name=metadata["name"],
            description=metadata["description"],
            trigger=str(metadata.get("trigger", "") or ""),
            enabled=metadata.get("enabled", True),
            content=body.strip(),
            file_path=file_path,
            relative_path="./" + os.path.relpath(file_path, self.root_dir).replace("\\", "/"),
        )

    def inspect_skill_content(self, content: str) -> Optional[Dict[str, Any]]:
        """Inspect in-memory skill content before it is written to disk."""
        split_result = self._split_skill_content(content)
        if not split_result:
            return None

        metadata, body = split_result
        return {
            "metadata": metadata,
            "body": body.strip(),
            "is_valid": self._validate_skill_metadata(metadata) is None,
            "validation_error": self._validate_skill_metadata(metadata),
        }

    def slugify_skill_name(self, name: str) -> str:
        """Build a safe folder name for a skill."""
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", name.strip()).strip("-").lower()
        return slug or "unnamed-skill"

    def build_skill_relative_path(self, skill_name: str) -> str:
        """Get the canonical project-relative path for a skill file."""
        return f"skills/{self.slugify_skill_name(skill_name)}/SKILL.md"

    def get_enabled_skills(self) -> List[Skill]:
        """Get list of enabled skills"""
        return [skill for skill in self.skills if skill.enabled]

    def get_skill_by_name(self, name: str) -> Optional[Skill]:
        """Get skill by name"""
        for skill in self.skills:
            if skill.name == name:
                return skill
        return None

    def match_trigger(self, message: str) -> Optional[Skill]:
        """
        Check if message matches any skill trigger pattern

        Args:
            message: User message to check

        Returns:
            Matched Skill or None
        """
        lowered_message = message.lower()

        for skill in self.get_enabled_skills():
            if skill.trigger.strip():
                patterns = skill.trigger.split("|")

                for pattern in patterns:
                    pattern = pattern.strip()
                    if pattern and re.search(pattern, message, re.IGNORECASE):
                        return skill

            if skill.name.lower() in lowered_message:
                return skill

        return None

    def get_skills_summary(self) -> str:
        """
        Generate a summary of all enabled skills for system prompt

        Returns:
            Formatted string listing all skills
        """
        return self.build_skills_prompt()

    def build_skills_snapshot(self) -> str:
        """Generate the SKILLS_SNAPSHOT.md content used in the system prompt."""
        enabled_skills = self.get_enabled_skills()
        lines = ["<available_skills>"]

        for skill in enabled_skills:
            lines.extend(
                [
                    "<skill>",
                    f"<name>{skill.name}</name>",
                    f"<description>{skill.description}</description>",
                    f"<location>{skill.relative_path}</location>",
                    "</skill>",
                ]
            )

        lines.append("</available_skills>")
        return "\n".join(lines)

    def build_skills_prompt(self) -> str:
        """Build the prompt-visible skills summary."""
        enabled_skills = self.get_enabled_skills()
        if not enabled_skills:
            return "No skills currently available."

        lines = ["Available skills:"]
        for skill in enabled_skills:
            lines.append(f"- {skill.name}: {skill.description}")

        return "\n".join(lines)

    def write_snapshot(self, workspace_dir: str) -> str:
        """Persist the skills snapshot to the workspace directory."""
        snapshot = self.build_skills_snapshot()
        snapshot_path = os.path.join(workspace_dir, "SKILLS_SNAPSHOT.md")
        with open(snapshot_path, "w", encoding="utf-8") as f:
            f.write(snapshot)
        return snapshot

    def reload_skills(self, verbose: bool = False):
        """Reload all skills from disk"""
        self.skills = []
        self._load_skills(verbose=verbose)

    def update_skill(self, name: str, content: str) -> bool:
        """
        Update skill content and save to disk

        Args:
            name: Skill name
            content: New SKILL.md content

        Returns:
            True if successful, False otherwise
        """
        skill = self.get_skill_by_name(name)
        if not skill:
            return False

        try:
            with open(skill.file_path, "w", encoding="utf-8") as f:
                f.write(content)

            # Reload skills
            self.reload_skills(verbose=False)
            return True
        except Exception as e:
            print(f"Error updating skill {name}: {e}")
            return False

    def get_all_skills_info(self) -> List[Dict]:
        """
        Get information about all skills for API response

        Returns:
            List of skill metadata dictionaries
        """
        return [
            {
                "name": skill.name,
                "description": skill.description,
                "trigger": skill.trigger,
                "enabled": skill.enabled,
                "location": skill.relative_path,
            }
            for skill in self.skills
        ]
