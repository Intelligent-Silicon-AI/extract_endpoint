from sqlalchemy import Column, Text, String, JSON, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import func
from .db import Base

class MultiJobs(Base):
    __tablename__ = 'multijobs'

    jobid = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    entry_point_url = Column(String, nullable=False)
    status = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())


class Evals(Base):
    __tablename__ = 'evals'
    jobid = Column(UUID(as_uuid=True), primary_key=True)
    train_loss = Column(String, nullable=False)
    test_loss = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())

class Run_URLs(Base):
    __tablename__ = 'runurls'
    runid = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()) 
    jobid = Column(UUID(as_uuid=True), nullable=False, foreign_key='multijobs.jobid')
    runurl = Column(String, nullable=False)
    runtype = Column(String, nullable=False)
    status = Column(String, nullable=False)
    artifact_url = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())
