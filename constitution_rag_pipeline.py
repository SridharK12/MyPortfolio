"""
Indian Constitution RAG Pipeline with Article-Based Chunking

This module combines intelligent article-based chunking with LangChain's
RAG pipeline for querying the Indian Constitution.

Features:
- Article-based chunking (from constitution_rag_chunker.py)
- LangChain RAG pipeline (from RAG.py)
- Token usage tracking
- Source document retrieval
- Persistent vector store with ChromaDB
"""

import re
import json
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks import get_openai_callback


# =========================
# CHUNKING LOGIC
# =========================

@dataclass
class ConstitutionChunk:
    """Represents a single chunk of the Constitution."""
    text: str
    metadata: Dict[str, any]
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_langchain_document(self) -> Document:
        """Convert to LangChain Document."""
        return Document(
            page_content=self.text,
            metadata=self.metadata
        )


class ConstitutionChunker:
    """
    Chunks the Indian Constitution based on articles with intelligent
    sub-chunking for long articles.
    """
    
    def __init__(self, max_chunk_size: int = 2000):
        """
        Initialize the chunker.
        
        Args:
            max_chunk_size: Maximum characters per chunk (default: 2000)
        """
        self.max_chunk_size = max_chunk_size
        self.chunks: List[ConstitutionChunk] = []
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as string
        """
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def identify_structure(self, text: str) -> Dict[str, str]:
        """
        Identify Parts and their boundaries in the Constitution.
        
        Args:
            text: Full constitution text
            
        Returns:
            Dictionary mapping part names to their starting positions
        """
        part_pattern = r'PART\s+([IVXLCDM]+)\s*\n\s*([A-Z\s,\-]+?)(?:\n|ARTICLES?)'
        parts = {}
        
        for match in re.finditer(part_pattern, text, re.MULTILINE):
            part_number = match.group(1).strip()
            part_title = match.group(2).strip()
            parts[f"PART {part_number}"] = {
                'title': part_title,
                'start': match.start()
            }
        
        return parts
    
    def extract_articles(self, text: str) -> List[Dict[str, any]]:
        """
        Extract all articles from the constitution text.
        
        Args:
            text: Full constitution text
            
        Returns:
            List of dictionaries containing article information
        """
        articles = []
        
        # Find the start of actual articles (after table of contents)
        part_i_match = re.search(r'PART\s+I\s*\n\s*THE UNION AND ITS TERRITORY', text)
        if part_i_match:
            text = text[part_i_match.start():]
        
        # Pattern to match article headers with titles
        article_pattern = r'\s+(\d+[A-Z]?)\.\s+([^.—\n]+?)\.?—'
        
        # Find all articles
        matches = list(re.finditer(article_pattern, text))
        
        for i, match in enumerate(matches):
            article_number = match.group(1).strip()
            article_title = match.group(2).strip()
            start_pos = match.end()
            
            # Find the end position (start of next article or end of text)
            if i < len(matches) - 1:
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(text)
            
            # Extract article content
            article_text = text[start_pos:end_pos].strip()
            
            # Clean up footnotes and extra whitespace
            article_text = re.sub(r'\n\s*\d+\s*\[', '\n', article_text)
            article_text = re.sub(r'\n\s*\d+\s*\*+', '\n', article_text)
            article_text = article_text.replace('\f', '\n')
            article_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', article_text)
            article_text = re.sub(r' +', ' ', article_text)
            
            # Only include if there's substantial content
            if len(article_text) > 50:
                articles.append({
                    'number': article_number,
                    'title': article_title,
                    'text': article_text,
                    'full_text': f"Article {article_number}. {article_title}\n\n{article_text}"
                })
        
        return articles
    
    def determine_part(self, article_number: str) -> Optional[str]:
        """
        Determine which Part an article belongs to based on article numbering.
        
        Args:
            article_number: The article number
            
        Returns:
            Part designation or None
        """
        try:
            num = int(re.match(r'\d+', article_number).group())
            
            # Mapping based on Indian Constitution structure
            if num <= 4:
                return "PART I"
            elif num <= 11:
                return "PART II"
            elif num <= 35:
                return "PART III"
            elif num <= 51:
                return "PART IV"
            elif num <= 147:
                return "PART V"
            elif num <= 237:
                return "PART VI"
            elif num <= 242:
                return "PART VII"
            elif num <= 243:
                return "PART VIII"
            elif num <= 255:
                return "PART IX"
            elif num <= 300:
                return "PART X"
            else:
                return "PART XI+"
        except:
            return None
    
    def split_long_article(self, article: Dict[str, any]) -> List[str]:
        """
        Split a long article into smaller chunks while preserving context.
        
        Args:
            article: Article dictionary with text
            
        Returns:
            List of text chunks
        """
        text = article['full_text']
        
        if len(text) <= self.max_chunk_size:
            return [text]
        
        chunks = []
        
        # Try to split by paragraphs first
        paragraphs = text.split('\n\n')
        current_chunk = f"Article {article['number']}: {article['title']}\n\n"
        header_length = len(current_chunk)
        
        for para in paragraphs:
            # If adding this paragraph exceeds limit, save current chunk
            if len(current_chunk) + len(para) + 2 > self.max_chunk_size and len(current_chunk) > header_length:
                chunks.append(current_chunk.strip())
                current_chunk = f"Article {article['number']} (continued):\n\n{para}\n\n"
            else:
                current_chunk += para + "\n\n"
        
        # Add remaining content
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # If paragraphs are too long, split by sentences
        if any(len(chunk) > self.max_chunk_size for chunk in chunks):
            refined_chunks = []
            for chunk in chunks:
                if len(chunk) <= self.max_chunk_size:
                    refined_chunks.append(chunk)
                else:
                    # Split by sentences
                    sentences = re.split(r'(?<=[.!?])\s+', chunk)
                    current = f"Article {article['number']} (continued):\n\n"
                    
                    for sentence in sentences:
                        if len(current) + len(sentence) + 1 > self.max_chunk_size:
                            refined_chunks.append(current.strip())
                            current = f"Article {article['number']} (continued):\n\n{sentence} "
                        else:
                            current += sentence + " "
                    
                    if current.strip():
                        refined_chunks.append(current.strip())
            
            chunks = refined_chunks
        
        return chunks
    
    def chunk_constitution(self, pdf_path: str) -> List[ConstitutionChunk]:
        """
        Main method to chunk the entire constitution.
        
        Args:
            pdf_path: Path to the Constitution PDF
            
        Returns:
            List of ConstitutionChunk objects
        """
        # Extract text
        print("📄 Extracting text from PDF...")
        text = self.extract_text_from_pdf(pdf_path)
        
        # Identify structure
        print("🏛️  Identifying Parts...")
        parts = self.identify_structure(text)
        
        # Extract articles
        print("📋 Extracting articles...")
        articles = self.extract_articles(text)
        print(f"✓ Found {len(articles)} articles")
        
        # Process each article
        chunks = []
        for article in articles:
            article_chunks = self.split_long_article(article)
            part = self.determine_part(article['number'])
            
            for chunk_num, chunk_text in enumerate(article_chunks, start=1):
                metadata = {
                    'article_number': article['number'],
                    'article_title': article['title'],
                    'chunk_number': chunk_num,
                    'total_chunks': len(article_chunks),
                    'part': part,
                    'document': 'Constitution of India',
                    'char_count': len(chunk_text),
                    'source': 'Indian Constitution PDF'
                }
                
                chunks.append(ConstitutionChunk(
                    text=chunk_text,
                    metadata=metadata
                ))
        
        self.chunks = chunks
        print(f"✓ Created {len(chunks)} chunks")
        return chunks
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get statistics about the chunks.
        
        Returns:
            Dictionary with statistics
        """
        if not self.chunks:
            return {}
        
        chunk_sizes = [len(chunk.text) for chunk in self.chunks]
        articles_with_multiple_chunks = len([
            c for c in self.chunks if c.metadata['total_chunks'] > 1
        ])
        
        return {
            'total_chunks': len(self.chunks),
            'total_articles': len(set(c.metadata['article_number'] for c in self.chunks)),
            'articles_with_multiple_chunks': articles_with_multiple_chunks,
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'chunks_over_limit': len([s for s in chunk_sizes if s > self.max_chunk_size])
        }


# =========================
# RAG PIPELINE
# =========================

class ConstitutionRAG:
    """
    RAG pipeline for querying the Indian Constitution using article-based chunks.
    """
    
    def __init__(self, persist_dir: str = "chroma_db_constitution"):
        """
        Initialize the RAG pipeline.
        
        Args:
            persist_dir: Directory for ChromaDB persistence
        """
        self.persist_dir = persist_dir
        self.vectorstore = None
        self.chunker = ConstitutionChunker(max_chunk_size=2000)
        
    def build_vector_store(self, pdf_path: str) -> Chroma:
        """
        Build vector store from Constitution PDF using article-based chunking.
        
        Args:
            pdf_path: Path to the Constitution PDF
            
        Returns:
            ChromaDB vector store
        """
        print("\n" + "="*60)
        print("BUILDING VECTOR STORE")
        print("="*60)
        
        # Chunk the constitution
        constitution_chunks = self.chunker.chunk_constitution(pdf_path)
        
        # Convert to LangChain documents
        print("\n🔄 Converting to LangChain documents...")
        documents = [chunk.to_langchain_document() for chunk in constitution_chunks]
        
        # Display statistics
        stats = self.chunker.get_statistics()
        print("\n📊 Chunking Statistics:")
        for key, value in stats.items():
            print(f"   {key.replace('_', ' ').title()}: {value}")
        
        # Create embeddings
        print("\n🔢 Creating embeddings...")
        embedding = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )
        
        # Build vector store
        print(f"💾 Building ChromaDB vector store at {self.persist_dir}...")
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding,
            persist_directory=self.persist_dir
        )
        
        print("✓ Vector store built successfully!")
        return self.vectorstore
    
    def load_vector_store(self) -> Chroma:
        """
        Load existing vector store from disk.
        
        Returns:
            ChromaDB vector store
        """
        print("\n📂 Loading existing vector store...")
        
        embedding = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )
        
        self.vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=embedding
        )
        
        print("✓ Vector store loaded!")
        return self.vectorstore
    
    def format_docs(self, docs: List[Document]) -> str:
        """
        Format retrieved documents for context.
        
        Args:
            docs: List of LangChain documents
            
        Returns:
            Formatted string
        """
        return "\n\n".join(doc.page_content for doc in docs)
    
    def ask_question(
        self, 
        query: str, 
        k: int = 5,
        temperature: float = 0
    ) -> Tuple[str, List[Document], Dict]:
        """
        Query the Constitution using RAG.
        
        Args:
            query: Question to ask
            k: Number of documents to retrieve
            temperature: LLM temperature
            
        Returns:
            Tuple of (answer, source_documents, token_stats)
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not loaded. Call build_vector_store() or load_vector_store() first.")
        
        # LLM
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=temperature
        )
        
        # Retriever
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        
        # Prompt
        prompt = ChatPromptTemplate.from_template("""
You are an expert assistant on the Indian Constitution. Answer questions based ONLY on the provided context from the Constitution articles.

Context:
{context}

Question:
{question}

Instructions:
- Provide clear, accurate answers based on the context
- Cite specific article numbers when relevant
- If the answer requires multiple articles, synthesize them coherently
- If the answer is not in the context, say "This information is not found in the provided articles"
- Use constitutional terminology appropriately

Answer:
""")
        
        # LCEL Chain (LangChain Expression Language)
        rag_chain = (
            {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Invoke with callback to track tokens
        with get_openai_callback() as cb:
            answer = rag_chain.invoke(query)
            
            # Get source docs separately
            source_docs = retriever.invoke(query)
            
            # Token usage stats
            token_stats = {
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "total_tokens": cb.total_tokens,
                "total_cost": cb.total_cost,
                "successful_requests": cb.successful_requests
            }
        
        return answer, source_docs, token_stats
    
    def display_results(
        self, 
        query: str,
        answer: str, 
        source_docs: List[Document], 
        token_stats: Dict
    ):
        """
        Display query results in a formatted way.
        
        Args:
            query: The question asked
            answer: The RAG answer
            source_docs: Source documents retrieved
            token_stats: Token usage statistics
        """
        print("\n" + "="*60)
        print("QUERY")
        print("="*60)
        print(query)
        
        print("\n" + "="*60)
        print("ANSWER")
        print("="*60)
        print(answer)
        
        print("\n" + "="*60)
        print("TOKEN USAGE")
        print("="*60)
        print(f"Prompt Tokens: {token_stats['prompt_tokens']}")
        print(f"Completion Tokens: {token_stats['completion_tokens']}")
        print(f"Total Tokens: {token_stats['total_tokens']}")
        print(f"Total Cost (USD): ${token_stats['total_cost']:.6f}")
        print(f"Successful Requests: {token_stats['successful_requests']}")
        
        print("\n" + "="*60)
        print("SOURCE ARTICLES")
        print("="*60)
        for i, doc in enumerate(source_docs, 1):
            metadata = doc.metadata
            print(f"\n--- Source {i} ---")
            print(f"Article: {metadata.get('article_number', 'N/A')}")
            print(f"Title: {metadata.get('article_title', 'N/A')}")
            print(f"Part: {metadata.get('part', 'N/A')}")
            print(f"Chunk: {metadata.get('chunk_number', 'N/A')}/{metadata.get('total_chunks', 'N/A')}")
            print(f"\nContent Preview:")
            print(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)


# =========================
# MAIN EXECUTION
# =========================

def main():
    """
    Main execution function demonstrating the RAG pipeline.
    """
    # Configuration
    PDF_PATH = "C:\\Sridhar\\MLProjects\\constitution_rag_v7\\data\\IndianConstitution.pdf"
    PERSIST_DIR = "chroma_db_constitution"
    
    # Initialize RAG pipeline
    rag = ConstitutionRAG(persist_dir=PERSIST_DIR)
    
    # Check if vector store exists
    if os.path.exists(PERSIST_DIR):
        print("📂 Found existing vector store")
        rag.load_vector_store()
    else:
        print("🆕 Building new vector store")
        rag.build_vector_store(PDF_PATH)
    
    # Example queries
    queries = [
        """Given that 'Education' is in the Concurrent List (List III, Entry 25), 
        how does the implementation of Article 21A create tensions or coordination 
        challenges in India's federal structure? Consider both legislative competence 
        and financial responsibility.""",
        
        "What are the Fundamental Rights guaranteed under Part III of the Constitution?",
        
        "Explain the procedure for amendment of the Constitution as per Article 368.",
    ]
    
    # Process first query
    query = queries[0]
    
    print("\n" + "="*60)
    print("QUERYING THE CONSTITUTION")
    print("="*60)
    
    answer, docs, token_stats = rag.ask_question(query, k=5)
    rag.display_results(query, answer, docs, token_stats)
    
    # Interactive mode
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("Ask questions about the Constitution (or 'quit' to exit)")
    
    while True:
        user_query = input("\n❓ Your question: ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not user_query:
            continue
        
        answer, docs, token_stats = rag.ask_question(user_query, k=5)
        rag.display_results(user_query, answer, docs, token_stats)


if __name__ == "__main__":
    main()
