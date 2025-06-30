from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os
import ssl
load_dotenv()

connect_args = {}
#if os.environ.get("DATABASE_SSL_MODE", 'False').lower() == 'true':
#    connect_args["ssl_context"] = ssl.SSLContext()

db_engine = create_engine(
    os.environ.get("DATABASE_URL"),
    pool_pre_ping=True,
    pool_recycle=3600,
    connect_args=connect_args,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
