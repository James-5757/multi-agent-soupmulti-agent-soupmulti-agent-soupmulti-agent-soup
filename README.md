# 🧩 Multi-Agent Turtle Soup Solver  
**A multi-agent reasoning system for solving “Turtle Soup / Lateral Thinking Puzzles” using DeepSeek APIs**

本项目实现了一个**多智能体协作的海龟汤解谜系统**：

- 3 类不同风格的 AI 玩家  
- 1 个负责总结的 AI  
- 1 个掌握真相的出题者  
- 自动进行多轮提问、回答、推理与总结  
- 最终输出整场推理日志（含真相）到 Markdown 文件  

支持 DeepSeek 的 `deepseek-chat` 模型。

---

## 📁 项目结构

```bash
.
├── agent.py                # 主程序
├── README.md               # 项目说明



## 🚀 功能特点
✔ 多 Agent 协作

系统内置 5 个角色：

角色	职责
出题者（RiddleMaster）	掌握真相，用“是 / 不是 / 无关 / 不完全是”进行渐进式回答
逻辑侦探（Logic Detective）	偏重因果链推理，提出高价值问题
怀疑派（Skeptic）	专门抓漏洞、怀疑模糊点
发散派（Divergent Thinker）	从非常规角度提问
总结官（Summary Agent）	每轮总结事实、指路、不剧透
🧩 运行流程概述

程序运行流程如下：

初始化 API、模型、日志系统

加载题库并随机选一则海龟汤

创建各类角色 Agent

开始循环：

逻辑侦探提问 → 出题者回答

怀疑派提问 → 出题者回答

发散派提问 → 出题者回答

总结官总结

循环到达设定轮数后结束

自动将完整对话 + 真相写入 Markdown 日志文件


🛠️ 环境需求

Python 3.8+

### 依赖库：

openai
python-dotenv (可选)


你需要 DeepSeek API key：

export DEEPSEEK_API_KEY="your_api_key"

🔧 安装依赖
pip install openai python-dotenv

▶️ 运行
python agent.py


运行后，终端会显示多轮推理过程，程序结束后会自动在当前目录生成一份日志：

soup_log_题目名_时间戳.md


日志内容包含：

每轮的提问与回答

总结官的总结

最终完整真相


🧩 示例输出（片段）
第 2 轮
逻辑侦探：死者是否提前知道自己将会遇害？
出题者：不是。这个判断在故事中并不成立。
怀疑派：死者是否和凶手存在私下矛盾？
出题者：无关。

===== 本轮总结 =====
- 目前可确认死者并未预料危险
- 死亡动机与个人恩怨无直接关系
- 建议从“事件发生地点”和“死者的行为”继续追问
==================================
