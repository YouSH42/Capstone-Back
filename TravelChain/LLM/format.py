from pydantic import BaseModel

class Location(BaseModel):
    latitude: float
    longitude: float
    address: str
    
class Event(BaseModel):
    start_time: str
    end_time: str

class Place(BaseModel):
    place_id: int
    content_type_id: int
    title: str
    location: Location
    has_event: bool
    event: Event
    summary: str
    
class InputFormat(BaseModel):
    places_retrieved_by_RAG: list[Place]
    places_selected_by_user: list[Place]
    schedule_selected_by_user: list[Place]
    user_input: str
    
class OutputFormat(BaseModel):
    model_inference: str
    model_output: str
    recommended_places: list[Place]
    recommended_schedule: list[Place]

class Chat(BaseModel):
    chat_id: int
    user_input: str
    model_output: str
    time: str
    
class Context(BaseModel):
    chats: list[Chat]