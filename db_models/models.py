from sqlalchemy.orm import Mapped, mapped_column, declarative_base
from uuid import UUID, uuid4
from pgvector.sqlalchemy import Vector
from datetime import datetime
from sqlalchemy import func

DeclarativeBase = declarative_base()

class AdditionalData(DeclarativeBase):
    __tablename__ = 'uniq_face'
    
    time_created: Mapped[datetime] = mapped_column(default=datetime.now, server_default=func.now())
    id: Mapped[UUID] = mapped_column(
        primary_key=True,
        default=uuid4,
        server_default=func.gen_random_uuid(),
    )
    person_id: Mapped[int]
    person_vector: Mapped[Vector] = mapped_column(Vector())
    face_vector: Mapped[Vector] = mapped_column(Vector())
    face_detection_conf: Mapped[float]