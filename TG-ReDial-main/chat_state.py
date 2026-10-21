#!/usr/bin/env python3
"""Multi-turn conversational movie preference elicitation state machine."""

from __future__ import annotations
from typing import Any, Dict, List, Optional


QUESTION_SEQUENCE = [
    {
        "type": "QGEN_INTEREST",
        "question": "你通常喜欢什么类型的电影？",
        "choices": ["喜剧", "爱情", "科幻", "动作", "恐怖", "动画", "剧情", "悬疑"],
        "answer_template": "我喜欢{type}片",
    },
    {
        "type": "QGEN_HISTORY",
        "question": "最近看过哪些电影？",
        "answer_template": "我刚看过{type}",
    },
    {
        "type": "QGEN_GENRE",
        "question": "你最喜欢的剧情类型是什么？",
        "choices": ["喜剧", "爱情", "科幻", "动作", "恐怖", "动画", "剧情", "悬疑"],
        "answer_template": "我喜欢{type}",
    },
    {
        "type": "QGEN_MOOD",
        "question": "今天想看什么心情？",
        "choices": ["exciting", "chill", "adventure", "romantic", "thoughtful", "suspense"],
        "answer_template": "我想看{type}心情",
    },
    {
        "type": "QGEN_CONSTRAINTS",
        "question": "有什么特别的要求吗？（可选）",
        "answer_template": "我想{type}",
    },
    {
        "type": "QGEN_FINAL",
        "question": "还有什么想补充的吗？",
        "answer_template": "还有{type}",
    },
]


class ConversationState:
    """Manages multi-turn Q&A state across conversation turns."""

    def __init__(self, max_questions: int = 6):
        self.answers = {}
        self.completed = False
        self.question_count = 0
        self.questions_asked = []
        self.max_questions = max_questions
        self.current_question_type = None

    def reset(self):
        self.answers.clear()
        self.completed = False
        self.question_count = 0
        self.questions_asked.clear()
        self.current_question_type = None

    def should_ask(self, entropy, entropy_threshold):
        return (entropy > entropy_threshold and
                not self.completed and
                self.question_count < self.max_questions)

    def get_next_question(self):
        if self.completed:
            return None
        if self.question_count >= self.max_questions:
            self.completed = True
            return None
        q = QUESTION_SEQUENCE[self.question_count % len(QUESTION_SEQUENCE)]
        self.questions_asked.append(q["type"])
        self.current_question_type = q["type"]
        self.question_count += 1
        return {
            "type": q["type"],
            "question": q["question"],
            "choices": q.get("choices", []),
            "answer_template": q.get("answer_template", ""),
        }

    def record_answer(self, answer):
        if self.current_question_type is None:
            return
        self.answers[self.current_question_type] = answer

    def to_dict(self):
        return {
            "answers": self.answers,
            "completed": self.completed,
            "question_count": self.question_count,
            "questions_asked": self.questions_asked,
        }

    def summarize_answers(self):
        if not self.answers:
            return "No preferences recorded yet."
        return " | ".join([f"{k}: {v}" for k, v in self.answers.items()])

    @classmethod
    def from_dict(cls, d):
        state = cls()
        state.answers = d.get("answers", {})
        state.completed = d.get("completed", False)
        state.question_count = d.get("question_count", 0)
        state.questions_asked = d.get("questions_asked", [])
        state.current_question_type = d.get("current_question_type")
        return state


if __name__ == "__main__":
    state = ConversationState()
    print("Testing 6 questions:")
    for i in range(6):
        q = state.get_next_question()
        if q:
            print(f"  Q{i+1}: {q['question']}")
            state.record_answer(f"answer_{i}")
    print(f"Completed: {state.completed}")
    print(f"Total questions: {state.question_count}")

