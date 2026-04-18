# 操作指南

## 技能调用协议 (SKILL PROTOCOL)
Prompt 中只会注入技能摘要，也就是每个 skill 的名称和描述，不会直接注入 skill 的完整正文。
当你要使用某个技能时，必须严格遵守以下步骤：

1. 你的第一步行动永远是先使用 `read_file` 读取 `workspace/SKILLS_SNAPSHOT.md`，找到目标 skill 的真实文件位置。
2. 然后再使用 `read_file` 读取对应的 `SKILL.md` Markdown 文件。
3. 仔细阅读文件中的内容、步骤和示例。
4. 根据文件中的指示，结合你内置的 Core Tools (`terminal`, `python_repl`, `fetch_url`) 来执行具体任务。

禁止直接猜测技能的参数或用法，必须先读取文件。

## 记忆协议

1. 长期事实记录在 `workspace/memory/MEMORY.md`。
2. 会话历史保存在 `workspace/sessions/*.json`。
3. 如需引用项目规则或用户画像，请优先参考工作区文件。
4. 如果你不确定某个事实是否准确，应明确说明不确定，而不是把猜测写进记忆。
