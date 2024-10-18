Retrieval model to efficiently search and retrieve pertinent information from vector store, subsequently integrating llmware BLING models to generate tailored responses.

## Directory Structure
.
├── Data/                      # Folder containing the PDF file
├── stores/                    # Folder where the vector store is persisted
├── Script/                    # Folder for Python scripts
    ├──ingest.py               # Script for ingesting documents into vector store
    ├──retrival.py             # script to interact with the vector store and retrieve document-based responses
├── app.py                     # Main application script                 
├── requirements.txt           # Python dependencies
└── README.md                  # This file
