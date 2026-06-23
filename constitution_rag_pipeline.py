"""
Indian Constitution RAG Pipeline with Optional Query Rewriting Layer

This module extends the original RAG pipeline with an OPTIONAL query rewriting layer
that uses gpt-4o-mini to improve retrieval quality. Users can choose to enable/disable
query rewriting to compare results.

Features:
- Optional query rewriting with gpt-4o-mini (smallest OpenAI model)
- Article-based chunking
- LangChain RAG pipeline
- Token usage tracking with separate rewriter costs
- Source document retrieval
- Persistent vector store with ChromaDB
- Side-by-side comparison mode
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
                            if current.strip():
                                refined_chunks.append(current.strip())
                            current = f"Article {article['number']} (continued):\n\n{sentence} "
                        else:
                            current += sentence + " "
                    
                    if current.strip():
                        refined_chunks.append(current.strip())
            
            return refined_chunks
        
        return chunks
    
    def chunk_constitution(self, pdf_path: str) -> List[ConstitutionChunk]:
        """
        Main method to chunk the entire constitution.
        
        Args:
            pdf_path: Path to the Constitution PDF
            
        Returns:
            List of ConstitutionChunk objects
        """
        print("📖 Extracting text from PDF...")
        text = self.extract_text_from_pdf(pdf_path)
        
        print("🔍 Extracting articles...")
        articles = self.extract_articles(text)
        print(f"   Found {len(articles)} articles")
        
        print("✂️  Chunking articles...")
        for article in articles:
            chunks = self.split_long_article(article)
            part = self.determine_part(article['number'])
            
            for i, chunk_text in enumerate(chunks, 1):
                chunk = ConstitutionChunk(
                    text=chunk_text,
                    metadata={
                        'article_number': article['number'],
                        'article_title': article['title'],
                        'part': part,
                        'chunk_number': i,
                        'total_chunks': len(chunks),
                        'chunk_size': len(chunk_text)
                    }
                )
                self.chunks.append(chunk)
        
        print(f"✓ Created {len(self.chunks)} chunks")
        return self.chunks
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the chunks.
        
        Returns:
            Dictionary with chunking statistics
        """
        if not self.chunks:
            return {}
        
        chunk_sizes = [len(c.text) for c in self.chunks]
        
        return {
            'total_chunks': len(self.chunks),
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'total_articles': len(set(c.metadata['article_number'] for c in self.chunks))
        }


# =========================
# QUERY REWRITER
# =========================

class QueryRewriter:
    """
    Rewrites user queries to improve retrieval quality using gpt-4o-mini.
    """
    
    def __init__(self):
        """Initialize the query rewriter with gpt-4o-mini."""
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",  # Smallest OpenAI model
            temperature=0
        )
        
        self.rewrite_prompt = ChatPromptTemplate.from_template("""
You are an expert at reformulating legal queries about the Indian Constitution to improve document retrieval.

Your task is to rewrite the user's query into an optimized search query that will better retrieve relevant constitutional articles.

Original Query: {query}

Guidelines:
1. Extract key legal concepts and constitutional terminology
2. Expand abbreviations (e.g., "FR" → "Fundamental Rights")
3. Add relevant synonyms for important terms
4. Make implicit references explicit (e.g., "voting age" → "right to vote age requirement")
5. For multi-part questions, focus on the core constitutional concepts
6. Remove conversational elements and keep only search-relevant terms
7. Include relevant article numbers if mentioned or implied
8. Add relevant Part names if identifiable (e.g., Part III for Fundamental Rights)

Output ONLY the rewritten query without any explanation or preamble.

Rewritten Query:""")
    
    def rewrite_query(self, query: str) -> Tuple[str, Dict]:
        """
        Rewrite a query to improve retrieval.
        
        Args:
            query: Original user query
            
        Returns:
            Tuple of (rewritten_query, token_stats)
        """
        chain = self.rewrite_prompt | self.llm | StrOutputParser()
        
        with get_openai_callback() as cb:
            rewritten = chain.invoke({"query": query})
            
            token_stats = {
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "total_tokens": cb.total_tokens,
                "total_cost": cb.total_cost
            }
        
        return rewritten.strip(), token_stats


# =========================
# RAG PIPELINE
# =========================

class ConstitutionRAG:
    """
    RAG pipeline for querying the Indian Constitution with optional query rewriting.
    """
    
    def __init__(self, persist_dir: str = "chroma_db"):
        """
        Initialize the RAG pipeline.
        
        Args:
            persist_dir: Directory to persist ChromaDB
        """
        self.persist_dir = persist_dir
        self.chunker = ConstitutionChunker()
        self.query_rewriter = QueryRewriter()
        self.vectorstore = None
        
    def build_vector_store(self, pdf_path: str) -> Chroma:
        """
        Build vector store from Constitution PDF.
        
        Args:
            pdf_path: Path to Constitution PDF
            
        Returns:
            ChromaDB vector store
        """
        print("\n🏗️  BUILDING VECTOR STORE")
        print("="*60)
        
        # Chunk the constitution
        chunks = self.chunker.chunk_constitution(pdf_path)
        
        # Convert to LangChain documents
        documents = [chunk.to_langchain_document() for chunk in chunks]
        
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
        
        # Build vector store with cosine similarity
        print(f"💾 Building ChromaDB vector store at {self.persist_dir} with cosine similarity...")
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding,
            persist_directory=self.persist_dir,
            collection_metadata={"hnsw:space": "cosine"}  # Explicitly use cosine similarity
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
            embedding_function=embedding,
            collection_metadata={"hnsw:space": "cosine"}  # Explicitly use cosine similarity
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
        temperature: float = 0,
        use_query_rewrite: bool = False
    ) -> Tuple[str, List[Document], Dict, Optional[str]]:
        """
        Query the Constitution using RAG with optional query rewriting.
        
        Args:
            query: Question to ask
            k: Number of documents to retrieve
            temperature: LLM temperature
            use_query_rewrite: Whether to use query rewriting (default: False)
            
        Returns:
            Tuple of (answer, source_documents, token_stats, rewritten_query)
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not loaded. Call build_vector_store() or load_vector_store() first.")
        
        # Initialize token stats
        rewriter_stats = {}
        rewritten_query = None
        search_query = query
        
        # Optional query rewriting
        if use_query_rewrite:
            print("\n🔄 Rewriting query...")
            rewritten_query, rewriter_stats = self.query_rewriter.rewrite_query(query)
            search_query = rewritten_query
            print(f"   Original: {query}")
            print(f"   Rewritten: {rewritten_query}")
        
        # LLM for answer generation
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=temperature
        )
        
        # Retriever (uses rewritten query if enabled)
        # Note: We'll use similarity_search_with_score to get scores
        
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
        
        # Retrieve documents with similarity scores
        docs_with_scores = self.vectorstore.similarity_search_with_score(search_query, k=k)
        
        # Extract just the documents for the RAG chain
        source_docs = [doc for doc, score in docs_with_scores]
        
        # Add similarity scores to document metadata
        for (doc, score), original_doc in zip(docs_with_scores, source_docs):
            original_doc.metadata['similarity_score'] = float(score)
        
        # LCEL Chain (LangChain Expression Language)
        rag_chain = (
            {"context": lambda _: self.format_docs(source_docs), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Invoke with callback to track tokens
        with get_openai_callback() as cb:
            answer = rag_chain.invoke(search_query)
            
            # Token usage stats for RAG
            rag_stats = {
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "total_tokens": cb.total_tokens,
                "total_cost": cb.total_cost,
                "successful_requests": cb.successful_requests
            }
        
        # Combine token stats
        token_stats = {
            "rag": rag_stats,
            "rewriter": rewriter_stats,
            "total_cost": rag_stats["total_cost"] + rewriter_stats.get("total_cost", 0)
        }
        
        return answer, source_docs, token_stats, rewritten_query
    
    def display_results(
        self, 
        query: str,
        answer: str, 
        source_docs: List[Document], 
        token_stats: Dict,
        rewritten_query: Optional[str] = None,
        use_query_rewrite: bool = False
    ):
        """
        Display query results in a formatted way.
        
        Args:
            query: The original question asked
            answer: The RAG answer
            source_docs: Source documents retrieved
            token_stats: Token usage statistics
            rewritten_query: The rewritten query (if query rewriting was used)
            use_query_rewrite: Whether query rewriting was used
        """
        print("\n" + "="*60)
        print("QUERY")
        print("="*60)
        print(f"Original: {query}")
        if use_query_rewrite and rewritten_query:
            print(f"Rewritten: {rewritten_query}")
        
        print("\n" + "="*60)
        print("ANSWER")
        print("="*60)
        print(answer)
        
        print("\n" + "="*60)
        print("TOKEN USAGE")
        print("="*60)
        
        if use_query_rewrite and "rewriter" in token_stats:
            print("Query Rewriter (gpt-4o-mini):")
            print(f"  Prompt Tokens: {token_stats['rewriter']['prompt_tokens']}")
            print(f"  Completion Tokens: {token_stats['rewriter']['completion_tokens']}")
            print(f"  Total Tokens: {token_stats['rewriter']['total_tokens']}")
            print(f"  Cost: ${token_stats['rewriter']['total_cost']:.6f}")
            print()
        
        print("RAG Pipeline (gpt-4o-mini):")
        print(f"  Prompt Tokens: {token_stats['rag']['prompt_tokens']}")
        print(f"  Completion Tokens: {token_stats['rag']['completion_tokens']}")
        print(f"  Total Tokens: {token_stats['rag']['total_tokens']}")
        print(f"  Cost: ${token_stats['rag']['total_cost']:.6f}")
        print()
        print(f"Total Cost (USD): ${token_stats['total_cost']:.6f}")
        print(f"Successful Requests: {token_stats['rag']['successful_requests']}")
        
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
            
            # Display similarity score (lower is better for cosine distance)
            similarity_score = metadata.get('similarity_score', 'N/A')
            if similarity_score != 'N/A':
                # Convert distance to similarity (1 - distance for cosine)
                similarity_percentage = (1 - similarity_score) * 100
                print(f"Similarity Score: {similarity_percentage:.2f}% (distance: {similarity_score:.4f})")
            else:
                print(f"Similarity Score: {similarity_score}")
            
            print(f"\nContent Preview:")
            print(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
    
    def compare_with_and_without_rewrite(self, query: str, k: int = 5) -> Dict:
        """
        Compare results with and without query rewriting.
        
        Args:
            query: Question to ask
            k: Number of documents to retrieve
            
        Returns:
            Dictionary with both results for comparison
        """
        print("\n" + "="*80)
        print("COMPARISON MODE: WITH vs WITHOUT QUERY REWRITING")
        print("="*80)
        
        # Without query rewriting
        print("\n" + "🔵 " + "="*75)
        print("WITHOUT QUERY REWRITING")
        print("="*80)
        answer_no_rewrite, docs_no_rewrite, tokens_no_rewrite, _ = self.ask_question(
            query, k=k, use_query_rewrite=False
        )
        self.display_results(query, answer_no_rewrite, docs_no_rewrite, tokens_no_rewrite, 
                           use_query_rewrite=False)
        
        # With query rewriting
        print("\n\n" + "🟢 " + "="*75)
        print("WITH QUERY REWRITING")
        print("="*80)
        answer_rewrite, docs_rewrite, tokens_rewrite, rewritten = self.ask_question(
            query, k=k, use_query_rewrite=True
        )
        self.display_results(query, answer_rewrite, docs_rewrite, tokens_rewrite, 
                           rewritten, use_query_rewrite=True)
        
        # Summary comparison
        print("\n\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        
        # Compare retrieved articles
        articles_no_rewrite = set(doc.metadata['article_number'] for doc in docs_no_rewrite)
        articles_rewrite = set(doc.metadata['article_number'] for doc in docs_rewrite)
        
        print("\nRetrieved Articles:")
        print(f"  Without rewriting: {sorted(articles_no_rewrite)}")
        print(f"  With rewriting: {sorted(articles_rewrite)}")
        print(f"  Overlap: {sorted(articles_no_rewrite & articles_rewrite)}")
        print(f"  Only in no-rewrite: {sorted(articles_no_rewrite - articles_rewrite)}")
        print(f"  Only in rewrite: {sorted(articles_rewrite - articles_no_rewrite)}")
        
        print("\nCost Comparison:")
        print(f"  Without rewriting: ${tokens_no_rewrite['total_cost']:.6f}")
        print(f"  With rewriting: ${tokens_rewrite['total_cost']:.6f}")
        print(f"  Additional cost for rewriting: ${tokens_rewrite['total_cost'] - tokens_no_rewrite['total_cost']:.6f}")
        
        return {
            "no_rewrite": {
                "answer": answer_no_rewrite,
                "docs": docs_no_rewrite,
                "tokens": tokens_no_rewrite
            },
            "rewrite": {
                "answer": answer_rewrite,
                "docs": docs_rewrite,
                "tokens": tokens_rewrite,
                "rewritten_query": rewritten
            }
        }


# =========================
# MAIN EXECUTION
# =========================

def main():
    """
    Main execution function demonstrating the RAG pipeline with query rewriting options.
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
    
    # Example query
    query = """Given that 'Education' is in the Concurrent List (List III, Entry 25), 
    how does the implementation of Article 21A create tensions or coordination 
    challenges in India's federal structure? Consider both legislative competence 
    and financial responsibility."""
    
    print("\n" + "="*60)
    print("DEMO: QUERY REWRITING COMPARISON")
    print("="*60)
    
    # Compare both approaches
    rag.compare_with_and_without_rewrite(query, k=5)
    
    # Interactive mode
    print("\n\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("Ask questions about the Constitution")
    print("Commands:")
    print("  - Type your question to use WITHOUT query rewriting")
    print("  - Prefix with 'rewrite:' to use WITH query rewriting")
    print("  - Type 'compare:' to compare both approaches")
    print("  - Type 'quit' to exit")
    
    while True:
        user_input = input("\n❓ Your input: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Parse command
        if user_input.lower().startswith('compare:'):
            query = user_input[8:].strip()
            if query:
                rag.compare_with_and_without_rewrite(query, k=5)
        elif user_input.lower().startswith('rewrite:'):
            query = user_input[8:].strip()
            if query:
                answer, docs, tokens, rewritten = rag.ask_question(query, k=5, use_query_rewrite=True)
                rag.display_results(query, answer, docs, tokens, rewritten, use_query_rewrite=True)
        else:
            # Default: no rewriting
            answer, docs, tokens, _ = rag.ask_question(user_input, k=5, use_query_rewrite=False)
            rag.display_results(user_input, answer, docs, tokens, use_query_rewrite=False)


if __name__ == "__main__":
    main()
