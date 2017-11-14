from sqlalchemy import Column, Integer, String, Text, ForeignKey, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship


Base = declarative_base()

authorArticle_table = Table('authorArticle', Base.metadata,
    Column('author_id', Integer, ForeignKey('author.id')),
    Column('article_id', Integer, ForeignKey('article.id'))
)

class Author(Base):
    __tablename__ = 'author'

    id = Column(Integer, primary_key=True, autoincrement=True)
    firstName = Column(String(50))
    lastName = Column(String(50))
    category = Column(Text)
    articles = relationship(
        "Article",
        secondary=authorArticle_table,
        back_populates="authors"
    )

    def __repr__(self):
        """Show this object (database record)"""
        return "Author(%d, %s, %s)" % (
            self.id, self.firstName, self.lastName)

    @hybrid_property
    def fullName(self):
        return self.firstName + " " + self.lastName

    @hybrid_property
    def articleAbstracts(self):
        abstracts = [article.abstract for article in self.articles]
        return abstracts

class Article(Base):
    __tablename__ = 'article'

    id = Column(Integer, primary_key=True, autoincrement=True)
    category = Column(Text)
    title = Column(Text)
    abstract = Column(Text)
    authors = relationship(
        "Author",
        secondary=authorArticle_table,
        back_populates="articles"
    )

    def __repr__(self):
        """Show this object (database record)"""
        return "Author(%d, %s, %s)" % (
            self.id, self.firstName, self.lastName)