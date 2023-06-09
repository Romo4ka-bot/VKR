from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from app.config import DATABASE_URL

Base = declarative_base()
engine = create_engine(DATABASE_URL)


def connect_db():
    session = Session(bind=engine.connect())
    return session


class UserPrediction(Base):
    __tablename__ = "user_prediction"

    id = Column(Integer, primary_key=True, index=True)
    total_cholesterol = Column(Integer)
    created_at = Column(String, default=datetime.utcnow())
    user = Column(Integer, ForeignKey('user.id'))


class User(Base):
    __tablename__ = "account"

    id = Column(Integer, primary_key=True)
    first_name = Column(String)
    second_name = Column(String)
    patronymic = Column(String)
    email = Column(String)
    hashed_password = Column(String)
