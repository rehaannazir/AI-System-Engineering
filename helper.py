from pydantic import BaseModel
from typing import List

class MultiQuery(BaseModel):

    queries : List[str]
