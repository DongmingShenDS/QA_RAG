# QA_RAG
RAG based QA system based on domain knowledge from local PDFs, useful for reviewing for exams :)

## How to Use (Testing Phase)

1. Prepare a folder containing the PDFs relevant to your exam material.
2. Update the chat.read_directory("pdfs") in the script to point to your specific PDF directory.
3. Pose questions to the system as needed.

## Updates
- RAG based on contents from Neo4j knowledge graph + Chroma vector database
- TODO: Query-translation (pre-process query for better retrival results)
