from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config import DATABASE_URL
from app.model import Base, engine


def main():
    # session = Session(bind=engine.connect())
    Base.metadata.create_all(engine)

    # session.execute("""create table user_prediction (
    # id integer not null primary key,
    # total_cholesterol integer,
    # created_at varchar(256),
    # user_id integer
    # );""")

    if __name__ == '__main__':
        main()
