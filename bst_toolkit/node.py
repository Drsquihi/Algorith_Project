from dataclasses import dataclass, field
from typing import Optional
@dataclass
class TrialNode:
    """
    A single node in the hyperparameter BST.
    The BST is keyed by `score`.
    Attributes
    Algorithmic Workshop — Final Project 8
    ----------
    score : the evaluation metric for this trial (e.g. accur
    acy)
    params : the hyperparameter dictionary used in this trial
    left : left child node (score strictly less than this n
    ode)
    right : right child node (score strictly greater than th
    is node)
    """
    score: float
    params: dict
    left: Optional["TrialNode"] = field(default=None, repr=False)
    right: Optional["TrialNode"] = field(default=None, repr=False)
    def __lt__(self, other: "TrialNode") -> bool:
        return self.score < other.score
    
    def __repr__(self) -> str:
        return f"TrialNode(score={self.score}, params={self.params})"