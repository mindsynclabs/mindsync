import os
from pathlib import Path
import lancedb
from lancedb.pydantic import LanceModel, Vector
from pydantic import Field

# Define local database storage URI following our local-first architecture
DB_URI = os.getenv("LANCEDB_URI", "data/lancedb")

class DocumentChunk(LanceModel):
    """Schema definition for an embedded document chunk stored in LanceDB."""
    id: str
    text: str
    # Vector dimension size (e.g., 768 for common local embedding models like nomic-embed-text)
    vector: Vector(768) 
    metadata: str = "{}"

class VectorStoreManager:
    """Manages local connection and operations for LanceDB."""
    def __init__(self, uri: str = DB_URI):
        Path(uri).mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(uri)
        
    def get_or_create_table(self, table_name: str = "knowledge_base"):
        """Retrieve an existing table or create a new one based on the DocumentChunk schema."""
        if table_name in self.db.table_names():
            return self.db.open_table(table_name)
        else:
            # Create a new table using the Pydantic LanceModel schema
            return self.db.create_table(table_name, schema=DocumentChunk)

# Initialize a shared vector store instance for the module
vector_store = VectorStoreManager()