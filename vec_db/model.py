
import datetime
from typing import List

from pydantic import BaseModel


class Knowledge(BaseModel):
    keyword: List[str]
    content: str
    create_time: datetime.datetime
    creator_name: str
    certainty: float