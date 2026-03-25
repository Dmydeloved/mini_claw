"""
Agent Skills System for Mini-OpenClaw
Handles skill loading, parsing, and snapshot generation.
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

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
        self._load_skills()

    def _load_skills(self):
        """Scan skills directory and load all SKILL.md files"""
        if not os.path.exists(self.skills_dir):
            os.makedirs(self.skills_dir)
            print(f"Created skills directory: {self.skills_dir}")
            return

        for skill_folder in os.listdir(self.skills_dir):
            skill_path = os.path.join(self.skills_dir, skill_folder)

            # Skip if not a directory
            if not os.path.isdir(skill_path):
                continue

            # Look for SKILL.md
            skill_file = os.path.join(skill_path, "SKILL.md")
            if not os.path.exists(skill_file):
                print(f"Warning: No SKILL.md found in {skill_folder}")
                continue

            try:
                skill = self._parse_skill_file(skill_file)
                if skill:
                    self.skills.append(skill)
                    print(f"Loaded skill: {skill.name}")
            except Exception as e:
                print(f"Error loading skill from {skill_file}: {e}")

    def _parse_skill_file(self, file_path: str) -> Optional[Skill]:
        """Parse SKILL.md file with YAML frontmatter"""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract frontmatter
        frontmatter_pattern = r"^---\s*\n(.*?)\n---\s*\n(.*)$"
        match = re.match(frontmatter_pattern, content, re.DOTALL)

        if not match:
            print(f"Warning: No valid frontmatter in {file_path}")
            return None

        frontmatter_text = match.group(1)
        body = match.group(2)

        # Parse YAML frontmatter
        try:
            metadata = yaml.safe_load(frontmatter_text)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML in {file_path}: {e}")
            return None

        # Validate required fields
        required_fields = ["name", "description", "trigger"]
        for field in required_fields:
            if field not in metadata:
                print(f"Missing required field '{field}' in {file_path}")
                return None

        return Skill(
            name=metadata["name"],
            description=metadata["description"],
            trigger=metadata["trigger"],
            enabled=metadata.get("enabled", True),
            content=body.strip(),
            file_path=file_path,
            relative_path="./" + os.path.relpath(file_path, self.root_dir).replace("\\", "/"),
        )

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
        for skill in self.get_enabled_skills():
            # Split trigger patterns by |
            patterns = skill.trigger.split("|")

            for pattern in patterns:
                pattern = pattern.strip()
                # Case-insensitive regex match
                if re.search(pattern, message, re.IGNORECASE):
                    return skill

        return None

    def get_skills_summary(self) -> str:
        """
        Generate a summary of all enabled skills for system prompt

        Returns:
            Formatted string listing all skills
        """
        enabled_skills = self.get_enabled_skills()

        if not enabled_skills:
            return "No skills currently available."

        summary_lines = ["Available Skills:"]
        for skill in enabled_skills:
            summary_lines.append(f"- {skill.name}: {skill.description}")

        return "\n".join(summary_lines)

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

    def write_snapshot(self, workspace_dir: str) -> str:
        """Persist the skills snapshot to the workspace directory."""
        snapshot = self.build_skills_snapshot()
        snapshot_path = os.path.join(workspace_dir, "SKILLS_SNAPSHOT.md")
        with open(snapshot_path, "w", encoding="utf-8") as f:
            f.write(snapshot)
        return snapshot

    def reload_skills(self):
        """Reload all skills from disk"""
        self.skills = []
        self._load_skills()

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
            self.reload_skills()
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
