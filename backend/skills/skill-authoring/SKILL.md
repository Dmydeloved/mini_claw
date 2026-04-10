---
name: skill-authoring
description: Create, add, save, update, or repair project skills from a user's natural-language request or partial SKILL.md draft. Use this when the user wants a new skill added to the project, wants an existing skill changed, or provides notes, API docs, examples, or a draft that should become a valid `skills/<skill_name>/SKILL.md`.
trigger: 添加skill|新增skill|创建skill|保存skill|更新skill|技能草稿|SKILL.md|add skill|create skill|save skill|update skill
enabled: true
---

# Skill Authoring

Use this skill when the task is to create or update a skill under `backend/skills`.

## Goal

Turn the user's request into a valid `SKILL.md`, write it to `skills/<slug>/SKILL.md`, and make sure the saved content can be loaded by the skill system.

## Workflow

1. Read `workspace/SKILLS_SNAPSHOT.md` when you need to inspect existing skills or confirm whether this is a new skill versus an update.
2. Determine the target skill name.
   - If the user already provided `name` in a draft, keep it.
   - If the user only described a capability, infer a concise, stable name that works well as a folder slug.
3. Normalize the content into a valid `SKILL.md`.
   - Frontmatter must include at least `name` and `description`.
   - Add `enabled: true` unless the user explicitly wants it disabled.
   - Add `trigger` only when routing hints are useful.
4. Write the result with `write_file` to `skills/<slug>/SKILL.md`.
5. Rely on the `write_file` tool result to confirm whether the skill loaded successfully.
6. In the final reply, briefly state whether the skill was created or updated and mention the saved path.

## Authoring Rules

- Keep the skill body focused on execution guidance for the model, not README-style documentation.
- The description should clearly say what the skill does and when it should be used, because only the name and description are injected before the skill is opened.
- If the user provides a partial draft, preserve their intent but repair the format and fill in only the missing pieces needed for a valid skill.
- If updating an existing skill, keep unchanged behavior intact unless the user asked for a behavioral change.
- Prefer concise instructions with concrete steps over long explanations.
- Make reasonable assumptions when details are missing, unless the missing detail would materially change the behavior.

## Minimal Skeleton

```md
---
name: example-skill
description: Explain what this skill does and when to use it.
trigger: optional|routing|hints
enabled: true
---

# Example Skill

## When to Use

Describe the requests that should trigger this skill.

## Steps

1. Explain how to interpret the user request.
2. Explain which tools or files to use.
3. Explain how to produce the final result.
```

## Notes

- Paths are relative to the backend project root.
- The canonical save location is always `skills/<slug>/SKILL.md`.
