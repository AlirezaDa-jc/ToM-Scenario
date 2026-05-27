from __future__ import annotations
 
from typing import Dict
from pydantic import BaseModel, Field
 
 
class BeliefState(BaseModel):
    object_locations: Dict[str, str] = Field(default_factory=dict)
 
    def update(self, obj: str, location: str) -> None:
        self.object_locations[obj] = location
 
    def get(self, obj: str) -> str:
        return self.object_locations.get(obj, "UNKNOWN")