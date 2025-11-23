"""
multi_agent_soup_deepseek.py

多智能体【海龟汤】示例：
- 使用 DeepSeek Chat API（OpenAI 兼容风格）
- 多个不同人格的 Agent 自动轮流提问
- 出题者 Agent 根据真相回答
- 支持题库随机出题
- 支持将整局对话记录保存为 markdown 日志
- 通用渐进泄露控制 + 提问去重 + 总结官只做总结
"""

import os
import re
import random
from dataclasses import dataclass
from typing import List, Tuple
from datetime import datetime
from openai import OpenAI

# ========= 1. DeepSeek 配置 =========
# 请先设置环境变量：DEEPSEEK_API_KEY
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not DEEPSEEK_API_KEY:
    raise RuntimeError("请先在环境变量中设置 DEEPSEEK_API_KEY")

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"  # DeepSeek 的 base_url
)

MODEL_NAME = "deepseek-chat"  # 也可以换成 deepseek-reasoner 等


# ========= 2. 海龟汤题目定义 & 题库 =========
@dataclass
class Riddle:
    title: str
    story: str
    truth: str


def build_riddle_bank() -> List[Riddle]:
    """
    构建一个海龟汤题库，你可以自行增加更多题目。
    """
    riddles: List[Riddle] = []

    # 题目 1：海边餐厅的男人（经典版）
    riddles.append(
        Riddle(
            title="海边餐厅的男人",
            story=(
                "一个男人走进海边餐厅，点了一份海龟汤。\n"
                "他只喝了一口，就脸色大变，随后离开餐厅并选择了自杀。\n"
                "请问，为什么？"
            ),
            truth=(
                "很久以前，这个男人曾在一次海难中幸存下来。\n"
                "当时他和其他幸存者被困在一座荒岛上，"
                "同伴们说给他吃的是海龟肉，这样他才活下来。\n"
                "多年后，他在海边餐厅第一次真正喝到海龟汤，"
                "发现味道与当年完全不同，于是意识到当年同伴骗了他，"
                "他当年吃的其实是同伴的人肉。\n"
                "在难以承受的震撼与自责中，他选择了自杀。"
            ),
        )
    )

    # 题目 2：沙漠中的黑衣人
    riddles.append(
        Riddle(
            title="沙漠中的黑衣人",
            story=(
                "一个穿着黑衣黑裤的男人死在沙漠中，"
                "旁边有一根半截的火柴，没有任何脚印。\n"
                "请问，他是怎么死的？"
            ),
            truth=(
                "这名男子原本在一架即将坠毁的飞机上。\n"
                "飞机燃料不足，机上几个人决定用抽签方式决定谁要跳伞。\n"
                "他们用火柴来抽长短签，这个男人抽到了最短的一根，"
                "被迫带着降落伞跳下飞机，最后死在沙漠中。\n"
                "旁边的半截火柴就是当时抽签时留下的证据。"
            ),
        )
    )

    # 题目 3：电梯里的男人
    riddles.append(
        Riddle(
            title="电梯里的男人",
            story=(
                "一个男人住在一栋大楼的 10 楼。\n"
                "每天出门时，他都会乘电梯到 1 楼；\n"
                "回来时，却只按 7 楼，然后走楼梯走到 10 楼。\n"
                "只有在下雨天或电梯里有别人的时候，他才会直接按 10 楼。\n"
                "请问，为什么？"
            ),
            truth=(
                "这个男人个子很矮，手臂够不到 10 楼的按钮，只能按到 7 楼，"
                "然后再走楼梯上去。\n"
                "在雨天时，他会带伞，就可以用伞去按 10 楼；\n"
                "有别人的时候，则可以拜托别人帮他按 10 楼。"
            ),
        )
    )

    # 你可以继续 append 更多题目……

    return riddles


def choose_random_riddle() -> Riddle:
    """
    随机选择一道题。
    """
    bank = build_riddle_bank()
    return random.choice(bank)


# ========= 3. 抽象 Agent 基类 =========
class BaseAgent:
    def __init__(self, name: str, persona: str):
        """
        persona: 这个 Agent 的人格设定（system prompt）
        """
        self.name = name
        self.persona = persona

    def chat(self, user_content: str) -> str:
        """
        调用 DeepSeek Chat Completions。
        适当降低 temperature，减少瞎编和剧透的冲动。
        """
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": self.persona},
                {"role": "user", "content": user_content},
            ],
            temperature=0.5,
        )
        return resp.choices[0].message.content.strip()


# ========= 4. 玩家 Agent：负责“提问” =========
class QuestionAgent(BaseAgent):
    """
    玩家 Agent：根据题目和历史问答生成新的问题。
    改进：
    - 对模型输出的问题做清洗，避免带角色名、引号等
    - 简单防止重复提问：如果这个玩家之前问过同样的问题，就换一个兜底问题
    """

    def _clean_question(self, raw: str) -> str:
        """
        对模型输出的问题做一些清洗：
        - 只保留第一行
        - 去掉类似“怀疑派：”“玩家：”之类前缀
        - 去掉引号和多余空白
        """
        q = raw.strip()
        if not q:
            return q

        # 只取第一行
        q = q.splitlines()[0].strip()

        # 去掉可能带上的角色前缀，例如“怀疑派：”“玩家：”
        pattern = rf"^(?:{re.escape(self.name)}|玩家|提问者)[：:\s]+"
        q = re.sub(pattern, "", q)

        # 去掉包裹的引号
        q = q.strip("「」\"' 　")

        return q

    def ask_question(
        self,
        riddle: Riddle,
        history: List[Tuple[str, str, str]],
    ) -> str:
        """
        根据题目和历史问答，生成一个新的【是/否】问题。
        history: [(who, question, answer), ...]
        """
        history_text = ""
        for who, q, a in history[-8:]:  # 只给最近几轮，防止太长
            history_text += f"{who}：{q}\n出题者：{a}\n"

        # 当前玩家已经问过的问题列表，用于简单去重
        previous_questions_by_me = [
            q for who, q, _ in history if who == self.name
        ]

        prompt = f"""
我们在玩“海龟汤”游戏，你是玩家：{self.name}。

题目：
《{riddle.title}》
{riddle.story}

以下是目前的部分提问与回答历史（可能为空）：
{history_text if history_text else "（暂无历史）"}

你的任务：
- 提出一个“是/否”问题，用来帮助逼近真相。
- 问题应该尽量高效，能排除尽可能多的可能。
- 不要复述题目，也不要解释，只输出问题本身。
- 不要直接问“真相是不是 XXX”“他是不是吃了人肉/自杀原因是……”这类一问就结束游戏的问题，
  而是循序渐进地缩小范围。
- **避免重复之前已经被任何玩家明确问过的问题，如果想追问某一条线索，请换一个更细的角度。**
- 当最近几轮的问题都集中在同一个主题（比如全部在问“身高”“电梯按钮”），
  请尝试从题目中尚未深入讨论的线索入手，例如环境（地点、天气）、角色关系、动机、时间顺序等。

请现在输出你要问的下一个问题（只输出问题本身，不要加引号）。
"""
        raw_question = self.chat(prompt)
        question = self._clean_question(raw_question)

        # 再做一个简单兜底：如果太短或者空，就返回一个保底问题
        if len(question) < 2:
            question = "这件事与他过去的某次重要经历有关吗？"
        # 如果这个玩家之前已经问过一模一样的问题，也换一个兜底问题
        if question in previous_questions_by_me:
            question = "这件事与题目中提到的特殊环境或条件（比如天气、地点或他身边的物品）有关吗？"

        return question


# ========= 5. 总结 Agent：负责阶段总结（不再直接猜真相） =========
class SummaryAgent(BaseAgent):
    def summarize(self, riddle: Riddle, history: List[Tuple[str, str, str]]) -> str:
        history_text = ""
        for who, q, a in history[-12:]:
            history_text += f"{who}：{q}\n出题者：{a}\n"

        prompt = f"""
我们在玩“海龟汤”游戏，你是总结官：{self.name}。

题目：
《{riddle.title}》
{riddle.story}

以下是最近一段的问答历史：
{history_text if history_text else "（暂无历史）"}

请你做两件事：
1. 用 2~4 条要点总结目前已经比较明确的重要信息（用条目列出）。
2. 指出目前玩家可能忽视的方向或关键点，可以给一些提示，
   比如建议他们关注：时间顺序、动机、身体条件、外部环境（天气、地点、道具）、
   角色之间的信息差或误会等。
   但不要直接说出完整真相，也不要写“真相是……”，也不要用过于具体的名词直接揭露核心原因。

请用简短中文回答。
"""
        return self.chat(prompt)


# ========= 6. 出题者 Agent：知道真相，只负责回答 =========
class RiddleMaster(BaseAgent):
    def __init__(self, name: str, riddle: Riddle):
        persona = f"""
你是海龟汤游戏的出题者，名字叫 {name}。

你非常清楚这道题目的“完整真相”，如下：
{riddle.truth}

游戏规则：
- 玩家会问你各种是/否问题。
- 你必须站在“知道真相”的角度，判断问题和真相之间的关系。
- 你需要用以下四种回答方式之一开头：
  1）“是。”    —— 明确肯定
  2）“不是。”  —— 明确否定
  3）“无关。”  —— 与真相无关或者过于细节，没意义
  4）“不完全是。” —— 部分正确，需要澄清

通用难度控制规则（适用于所有题目）：
- 在整个游戏过程中，你要尽量让玩家通过多轮提问逐步接近真相，而不是在前几轮就说得太具体。
- 你永远不要主动完整复述真相，只能根据问题的角度做局部的、含蓄的回答。

具体泄露节奏（和轮次有关）：
- **前几轮（大约第 1~2 轮）**：
  - 回答尽量抽象，用“与他的某种条件有关”“与当时的真实情况有关”这类模糊表达。
  - 避免给出任何非常具体的名词细节（例如“人肉”“伞”“身高不够”“抽签”等）。
- **中间几轮（大约第 3~4 轮）**：
  - 可以开始更明确地确认某些方向是“有帮助的”或“接近真相”，
    但仍尽量使用相对抽象的表达，例如“与他无法直接做到某件事有关”“与一次误会或信息差有关”。
- **后面轮次（第 5 轮及以后）**：
  - 可以在玩家问得非常接近时，才慢慢提到更具体的细节，
    但也尽量通过“不完全是。”来先部分确认，再补充一点点信息。
  - 仍然禁止你一次性把完整故事讲清楚。

在给出上述开头后，你可以用 1~2 句话做一个非常简短的解释，
但不能直接把完整真相说出来，也不要直接剧透。

你的回答风格：简洁、严谨、含蓄、不要废话。
"""
        super().__init__(name=name, persona=persona)
        self.riddle = riddle

    def answer(self, question: str, round_id: int) -> str:
        """
        根据当前轮次做“渐进泄露”提示。
        不改变模型逻辑，只是在 prompt 里告诉它：现在是早期/中期/后期。
        """
        if round_id <= 2:
            phase_instruction = (
                "现在是游戏的**前期轮次**，请尽量使用抽象和含蓄的方式回答，"
                "不要给出过于具体的细节名词，只做方向性的确认或否定。"
            )
        elif round_id <= 4:
            phase_instruction = (
                "现在是游戏的**中期轮次**，你可以比前几轮稍微具体一点，"
                "但仍然避免直接点名真相中的关键名词，只描述类别、关系或抽象特征。"
            )
        else:
            phase_instruction = (
                "现在是游戏的**后期轮次**，玩家已经问了很多问题，"
                "你可以在他们问得非常接近时，适当给出更具体的提示，"
                "但依然禁止你一次性完整讲出真相。"
            )

        prompt = f"""
玩家问的问题是：
“{question}”

当前是第 {round_id} 轮提问。
{phase_instruction}

请你根据你所掌握的真相来判断这个问题的答案。

请严格遵守格式要求：
- 以“是。”或“不是。”或“无关。”或“不完全是。”开头。
- 然后可以加 1~2 句话解释原因。
- 不要在回答里主动完整说出真相，只允许含蓄地、局部地补充信息。
"""
        return self.chat(prompt)


# ========= 7. 驱动一局游戏 & 日志记录 =========
def run_game(max_rounds: int = 4, log_to_file: bool = True):
    # 随机选题
    riddle = choose_random_riddle()

    # 日志列表（最后写入 .md）
    log_lines: List[str] = []

    def log_print(text: str = ""):
        print(text)
        log_lines.append(text)

    # 定义不同人格玩家
    logic_player = QuestionAgent(
        name="逻辑侦探",
        persona=(
            "你是一个冷静理性的推理玩家，擅长用最少的问题缩小可能范围，"
            "偏好问时间顺序、因果关系、动机、身份等高价值信息。"
            "问题简洁、直接、不绕弯。"
        ),
    )

    skeptic_player = QuestionAgent(
        name="怀疑派",
        persona=(
            "你是一个极度多疑的玩家，总是假设事情背后有阴谋或隐藏条件，"
            "喜欢针对之前回答中的模糊点提出问题。"
        ),
    )

    creative_player = QuestionAgent(
        name="发散派",
        persona=(
            "你是一个非常有想象力的玩家，擅长从非常规角度提问，"
            "例如误会、角色关系、环境细节、心理变化等。"
        ),
    )

    summary_agent = SummaryAgent(
        name="总结官",
        persona=(
            "你擅长阅读问答历史，总结目前已知信息，指出下一步推理方向，"
            "但不会直接说出完整真相。"
        ),
    )

    master = RiddleMaster(name="出题者", riddle=riddle)

    players: List[QuestionAgent] = [logic_player, skeptic_player, creative_player]
    history: List[Tuple[str, str, str]] = []

    header = "====== 海龟汤：多智能体示例（DeepSeek） ======"
    log_print(header)
    log_print("")
    log_print(f"题目：《{riddle.title}》")
    log_print(riddle.story)
    log_print("\n===============================\n")

    for round_id in range(1, max_rounds + 1):
        log_print(f"\n----- 第 {round_id} 轮提问 -----\n")
        for p in players:
            # 1. 玩家提出问题
            q = p.ask_question(riddle, history)
            if len(q) < 2:
                continue

            # 2. 出题者回答（带入当前轮次）
            a = master.answer(q, round_id=round_id)

            # 3. 记录历史
            history.append((p.name, q, a))

            # 4. 打印这一问
            log_print(f"{p.name}：{q}")
            log_print(f"{master.name}：{a}\n")

        # 每轮结束让总结官说两句
        log_print(f"===== 第 {round_id} 轮小结（{summary_agent.name}） =====")
        summary = summary_agent.summarize(riddle, history)
        log_print(summary)
        log_print("========================================\n")

    log_print("游戏结束。如果你是人类玩家，可以现在自己来猜真相～")
    log_print("\n（提示：在日志文件最后会附上真相，方便复盘。）")

    # ====== 写入日志文件 ======
    if log_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = "".join(c for c in riddle.title if c.isalnum())
        filename = f"soup_log_{safe_title}_{timestamp}.md"

        # 在日志末尾追加真相
        log_lines.append("\n---\n")
        log_lines.append(f"## 题目：《{riddle.title}》真相（仅供复盘）\n")
        log_lines.append(riddle.truth)

        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))

        print(f"\n📄 本局对话已保存为日志文件：{filename}")


if __name__ == "__main__":
    # 这里你可以调节轮数
    run_game(max_rounds=4, log_to_file=True)
