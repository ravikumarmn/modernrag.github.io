"""
Setup script for the ModernRAG package.
"""

from setuptools import setup, find_packages

setup(
    name="modernrag",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.3.27",
        "langchain-community>=0.3.29",
        "langchain-openai>=0.3.33",
        "langchain-pinecone>=0.2.12",
        "langchain-text-splitters>=0.1.0",
        "pinecone-client>=3.0.0",
        "PyMuPDF>=1.26.4",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "typing-extensions>=4.5.0",
        "asyncio>=3.4.3",
    ],
    author="ModernRAG Team",
    author_email="ravikumarnaduvin@gmail.com",
    description="A modern Retrieval-Augmented Generation (RAG) system",
    keywords="rag, vector-store, pinecone, langchain",
    python_requires=">=3.8",
)
