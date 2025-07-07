import os
import logfire
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from pydantic_ai.agent import Agent
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from src.agent import build_agent, answer_question, BotResponse
from src.mcp_agent import answer_mcp_question, MCPBotResponse

logfire.configure(service_name='api', environment='staging')

app = FastAPI(title='Math, Database and Pydantic AI API')
logfire.instrument_fastapi(app)

# database setup
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///./test.db')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autoflush=False, autocommit=False, bind=engine)
Base = declarative_base()
logfire.instrument_sqlalchemy(engine)
logfire.instrument_mcp()


# database model
class Item(Base):
    __tablename__ = 'items'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String)


Base.metadata.create_all(bind=engine)


class ItemCreate(BaseModel):
    name: str
    description: str


class ItemResponse(BaseModel):
    id: int
    name: str
    description: str

    class Config:
        from_attributes = True


class AgentQuery(BaseModel):
    question: str


class MCPQuery(BaseModel):
    question: str


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_agent() -> Agent[None, BotResponse]:
    return build_agent()


# endpoint 1: division
@app.get('/divide/{numerator}/{denominator}')
async def divide(numerator: float, denominator: float):
    """
    Divides the numerator by the denominator and returns the result.
    """
    result = numerator / denominator
    return {'result': result}


# endpoint 2: fibonacci
@app.get('/fibonacci/{n}')
async def fibonacci(n: int):
    """
    Calculates the nth number in the Fibonacci sequence.
    Raises an HTTPException if n is negative.
    """
    if n < 0:
        raise HTTPException(status_code=400, detail='Input must be a non-negative integer')

    if n <= 1:
        return {'result': n}

    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b

    return {'result': b}


@app.post('/items/', response_model=ItemResponse)
async def create_item(item: ItemCreate, db: Session = Depends(get_db)):
    """
    Creates a new item in the database.
    """
    db_item = Item(name=item.name, description=item.description)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item


@app.get('/items/', response_model=list[ItemResponse])
async def read_items(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Retrieves items from the database with pagination.
    """
    items = db.query(Item).offset(skip).limit(limit).all()
    return items


@app.get('/items/{item_id}', response_model=ItemResponse)
async def read_item(item_id: int, db: Session = Depends(get_db)):
    """
    Retrieves a specific item by ID.
    Raises an HTTPException if the item is not found.
    """
    item = db.query(Item).filter(Item.id == item_id).first()
    if item is None:
        raise HTTPException(status_code=404, detail='Item not')
    return item


@app.post('/agent/query', response_model=BotResponse)
async def query_agent(query: AgentQuery, agent: Agent[None, BotResponse] = Depends(get_agent)):
    """
    Queries the PydanticAI agent with a user question and returns the response.
    """
    logfire.info(f'Quering agent with question:{query.question}')
    response = await answer_question(agent, query.question)
    return response


@app.post('/mcp/query', response_model=MCPBotResponse)
async def query_mcp_agent(query: MCPQuery):
    """
    Queries the MCP-enabled PydanticAI agent with browser automation capabilities.
    """
    logfire.info(f'Querying MCP agent with question: {query.question}')
    response = await answer_mcp_question(query.question)
    return response


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8000)
