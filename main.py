import os
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_chroma import Chroma
from IPython.display import display, Markdown
from glob import glob
import getpass

if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass(
        "Enter your OpenAI API Key: "
    )
# initialize OpenAIEmbeddings
openai_embed_model = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=os.environ.get("OPENAI_API_KEY"))

# initilize llm model
llm_model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=os.environ.get("OPENAI_API_KEY"))

def generate_chunk_context(org_doc, chunk):
    chunk_prompt = """
                    Treat like an expert in analyzing the research papers. You have to generate a brief, relevant and
                     meaningful context for a chunk of text based on the following research paper.

                     Below is the research paper

                     <Research Paper>
                     {paper}
                     </Research Paper>

                     Here is the chunk we want to situate within the whole document:
                     <chunk>
                     {chunk}
                     </chunk>

                     Provide a concise context (3-4 sentences max) for this chunk,
                     considering the following guidelines:

                    - Give a short succinct context to situate this chunk within the overall document
                    for the purposes of improving search retrieval of the chunk.
                    - Answer only with the succinct context and nothing else.
                    - Context should be mentioned like 'Focuses on ....'
                    do not mention 'this chunk or section focuses on...'

                    Context:
                    """

    prompt_template = ChatPromptTemplate.from_template(chunk_prompt)

    agentic_chunk_chain = (
                            prompt_template
                            |
                            llm_model
                            |
                            StrOutputParser()
                            )

    context = agentic_chunk_chain.invoke({'paper':org_doc, 'chunk':chunk})

    return context


pdf_docs = []
import pymupdf
def create_pdf_contextual_chunks(pdf):

    # loading page
    print(f"loading pdf file: {pdf}")
    pdf_loader = PyMuPDFLoader(pdf)
    pdf_page = pdf_loader.load()

    # chunking page
    splitter = RecursiveCharacterTextSplitter(chunk_size=3500, chunk_overlap=0)
    pdf_chunks = splitter.split_documents(pdf_page)

    # generating contextual chunks
    original_pdf = '\n'.join([pdf.page_content for pdf in pdf_chunks])
    contextual_chunks = []
    for chunk in pdf_chunks:
        context = generate_chunk_context(original_pdf, chunk)
        contextual_chunks.append(Document(page_content=context+'\n'+chunk.page_content, metadata=chunk.metadata))

    return contextual_chunks


# loading and processing data
pdf_files = glob("./sample_data/*.pdf")

for pdf in pdf_files:
    pdf_docs.extend(create_pdf_contextual_chunks(pdf))

print(f"Total docs read: {len(pdf_docs)}")

chroma_db = Chroma(collection_name='RAG_System_db',
                   collection_metadata={"hnsw:space":"cosine"},
                   embedding_function=openai_embed_model,
                   persist_directory='./RAG_System_db')

# Add the documents to the ChromaDB
chroma_db.add_documents(pdf_docs)

sim_retriever = chroma_db.as_retriever(search_type="similarity",
                                       search_kwargs={"k":3})

def display_docs(docs):
    print(len(docs))
    for doc in docs:
        print('Metadata:', doc.metadata)
        print('Content Brief:')
        display(Markdown(doc.page_content[:1000]))
        print()


query = "What are the main components of a RAG model, and how do they interact?"
top_docs = sim_retriever.invoke(query)
display_docs(top_docs)

query = "What are the two sub-layers in each encoder layer of the Transformer model?"
top_docs = sim_retriever.invoke(query)
display_docs(top_docs)

query = 'Explain how positional encoding is implemented in Transformers and why it is necessary.'
top_docs = sim_retriever.invoke(query)
display_docs(top_docs)