from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    paper_id: str
    title: str
    section: str
    text: str

    @property
    def index_text(self) -> str:
        return f"{self.title}. {self.section}. {self.text}"


@dataclass
class Claim:
    text: str
    chunk_ids: list[str]


@dataclass
class Answer:
    question: str
    claims: list[Claim]
    abstained: bool
    retrieved: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)  # milliseconds
    trace: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        if self.abstained or not self.claims:
            return "INSUFFICIENT EVIDENCE"
        return " ".join(c.text for c in self.claims)

    @property
    def cited(self) -> set[str]:
        return {i for c in self.claims for i in c.chunk_ids} if not self.abstained else set()
