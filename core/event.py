from __future__ import annotations
 
from typing import List
from pydantic import BaseModel
 
 
class Event(BaseModel):
    actor: str
    action: str
    target_object: str
    from_location: str
    to_location: str
    visible_to: List[str]