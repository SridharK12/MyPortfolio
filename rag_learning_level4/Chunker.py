import os, json, uuid, hashlib, logging
from datetime import datetime
import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# === CONFIGURATION ===
pdf_path = r"I:\Sridhar\rag_learning_level4\Documents\jesc101.pdf"
output_json = r"I:\Sridhar\rag_learning_level4\docs\text_chunks\chemistry_ch1_with_local_images.json"
chroma_store_path = r"I:\Sridhar\rag_learning_level4\docs\chroma_store"
image_save_root = r"I:\Sridhar\rag_learning_level4\images"

# Ensure folders exist
os.makedirs(os.path.dirname(output_json), exist_ok=True)
os.makedirs(image_save_root, exist_ok=True)

# === LOGGING SETUP ===
log_file = r"I:\Sridhar\rag_learning_level4\docs\process_log.txt"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logging.info("🚀 Starting PDF ingestion with local image saving")

# === COMPONENTS ===
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", ".", " ", ""]
)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
db = chromadb.PersistentClient(path=chroma_store_path)
collection = db.get_or_create_collection("chemistry_rag")

# === CAPTION GENERATOR PLACEHOLDER ===
def generate_caption(image_obj=None):
    """Simple placeholder caption generator."""
    return "Image extracted from chemistry chapter showing a diagram or figure."

# === OPEN PDF ===
doc = fitz.open(pdf_path)
logging.info(f"Opened PDF: {pdf_path} | Total pages: {doc.page_count}")

all_chunks = []

# === MAIN LOOP: PROCESS EACH PAGE ===
for page_index, page in enumerate(doc):
    page_number = page_index + 1
    text = page.get_text("text").strip()
    blocks = page.get_text("blocks")

    logging.info(f"📄 Processing Page {page_number}")

    # === TEXT CHUNKS ===
    if text:
        for chunk in splitter.split_text(text):
            all_chunks.append({
                "type": "text",
                "page": page_number,
                "content": chunk.strip(),
                "hash": hashlib.sha256(chunk.encode("utf-8")).hexdigest()
            })

    # === TABLE DETECTION ===
    for b in blocks:
        x0, y0, x1, y1, block_text, *_ = b
        if "\t" in block_text or "|" in block_text or len(block_text.split("\n")) > 5:
            all_chunks.append({
                "type": "table",
                "page": page_number,
                "bbox": [x0, y0, x1, y1],
                "content": block_text.strip(),
                "hash": hashlib.sha256(block_text.encode("utf-8")).hexdigest()
            })

    # === EQUATION DETECTION ===
    for line in text.split("\n"):
        if any(sym in line for sym in ["=", "+", "-", "×", "÷", "∑", "∫", "√", "^", "_"]):
            eq = line.strip()
            all_chunks.append({
                "type": "equation",
                "page": page_number,
                "content": eq,
                "hash": hashlib.sha256(eq.encode("utf-8")).hexdigest()
            })

    # === IMAGE EXTRACTION ===
    image_list = page.get_images(full=True)
    for img_index, img in enumerate(image_list, start=1):
        xref = img[0]
        pix = fitz.Pixmap(doc, xref)

        # Create per-page folder
        page_folder = os.path.join(image_save_root, f"page_{page_number}")
        os.makedirs(page_folder, exist_ok=True)

        # Define image path
        img_filename = f"img_{img_index}.png"
        img_path = os.path.join(page_folder, img_filename)

        try:
            # Handle unsupported colorspaces (CMYK, alpha)
            if pix.colorspace is None or pix.n >= 4:
                pix_converted = fitz.Pixmap(fitz.csRGB, pix)
                pix_converted.save(img_path)
                pix_converted = None
            else:
                pix.save(img_path)
                print(f"✅ Saved: {img_path} ({pix.width}x{pix.height})")

            # Add metadata
            caption = generate_caption()
            image_hash = hashlib.sha256(f"{xref}-{pix.width}-{pix.height}".encode()).hexdigest()

            all_chunks.append({
                "type": "image",
                "page": page_number,
                "xref": xref,
                "width": pix.width,
                "height": pix.height,
                "local_path": img_path,
                "caption": caption,
                "hash": image_hash
            })

            logging.info(f"✅ Saved image page {page_number}-{img_index} → {img_path}")

        except Exception as e:
            logging.error(f"❌ Error saving image {img_index} on page {page_number}: {e}")

        finally:
            pix = None  # free memory

    logging.info(f"📄 Page {page_number} completed: {len(image_list)} images processed.")

doc.close()
logging.info(f"✅ PDF parsing complete. Total chunks collected: {len(all_chunks)}")

# === SAVE STRUCTURED DATA TO JSON ===
data = {
    "doc_id": os.path.basename(pdf_path),
    "timestamp": datetime.now().isoformat(),
    "chunks": all_chunks
}

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

logging.info(f"💾 Saved structured chunks with local image paths to {output_json}")
print(f"✅ Extraction complete. Total chunks: {len(all_chunks)}")

# === STORE EMBEDDINGS IN CHROMADB ===
text_like = [c for c in all_chunks if c["type"] in ["text", "table", "equation", "image"]]

for c in text_like:
    content_id = str(uuid.uuid4())

    # For images, embed the caption instead of binary data
    content = c["caption"] if c["type"] == "image" else c["content"]

    emb = embedding_model.encode(content)

    # 🧠 Build clean metadata (no None values)
    metadata = {
        "type": c["type"],
        "page": int(c["page"]),
        "hash": str(c["hash"])
    }

    # Add path only for images
    if "local_path" in c and c["local_path"]:
        metadata["local_path"] = str(c["local_path"])

    collection.add(
        ids=[content_id],
        embeddings=[emb],
        metadatas=[metadata],
        documents=[content]
    )

logging.info("✅ Stored embeddings (including image captions) in ChromaDB")
print("✅ Embeddings stored successfully.")
