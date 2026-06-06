print('RAG APPLICATION USING QWEN MODEL---')
print("HI")
metadata = {

    'dataset_id': '1234',

    'organization': 'Ajaz',

    'dataset_type': 's3',

}


from typing import List, Optional, Union, Dict, Sequence, Any, Type
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_community.document_loaders.text import TextLoader

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.html_bs import BSHTMLLoader
import logging as logger
from langchain_core.documents import Document


FILE_LOADER_TYPE = Union[Type[UnstructuredFileLoader], Type[TextLoader], Type[CSVLoader], Type[BSHTMLLoader]]



allsplit_doc_list = []

file_path = r"C:\Users\ajaza\OneDrive\Desktop\Resume"


def directory_loader(path: str,

                     glob: str = "**/[!.]*",

                     silent_errors: bool = False,

                     load_hidden: bool = False,

                     loader_cls: FILE_LOADER_TYPE = UnstructuredFileLoader,

                     loader_kwargs: Union[dict, None] = None,

                     recursive: bool = False,

                     show_progress: bool = False,

                     use_multithreading: bool = False,

                     max_concurrency: int = 4,

                     sample_size: int = 0,

                     randomize_sample: bool = False,

                     sample_seed: Union[int, None] = None):
    try:

        from langchain_community.document_loaders import DirectoryLoader

        loader = DirectoryLoader(path=path,

                                 glob=glob,

                                 silent_errors=silent_errors,

                                 load_hidden=load_hidden,

                                 loader_cls=loader_cls,

                                 loader_kwargs=loader_kwargs,

                                 recursive=recursive,

                                 show_progress=show_progress,

                                 use_multithreading=use_multithreading,

                                 max_concurrency=max_concurrency,

                                 sample_size=sample_size,

                                 randomize_sample=randomize_sample,

                                 sample_seed=sample_seed)

        return loader

    except Exception as e:

        logger.info('Exception in directory_loader as', e)

        raise e


def document_with_metadata(documents: List[Document], metadata) -> List[Document]:
    try:

        for index, doc in enumerate(documents):
            doc.metadata.update(metadata)  # Inject metadata directly into the document object

        return documents

    except Exception as e:

        logger.info('Exception in document_with_metadata as', e)

        raise e

from langchain_text_splitters import RecursiveCharacterTextSplitter

def splitdocs(docs, chunk_size=1000, chunk_overlap=200, separators=None, keep_separator=True):
    try:

        text_splitter  = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap,separators = separators,keep_separator=keep_separator)

        splitted_doc = text_splitter.split_documents(docs)

        return splitted_doc

    except Exception as e:

        logger.info('Exception in splitdocs as', e)

        raise e

def loader(file_path,metadata):

    loader = directory_loader(path=file_path,silent_errors=True,show_progress=True,loader_kwargs={'mode':'single'})

    documents = loader.load()

    final_document = document_with_metadata(documents, metadata)

    #final_document = documents

    allsplit_doc_list.extend(final_document)

    if allsplit_doc_list != None:
        final_splitted_document = splitdocs(allsplit_doc_list, chunk_size=1000,
                                                 chunk_overlap=200, separators=None,
                                                 keep_separator=True)

        all_split_docs = final_splitted_document
        print(allsplit_doc_list)
        print(all_split_docs)
        return all_split_docs

def embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print(embeddings)
    return embeddings
# vector = embeddings.embed_query("Open source embeddings are great")
# print(len(vector))  # 384

import os
def FaissDB(embedding_function, docs):
    try:

        #from langchain.vectorstores import FAISS
        from langchain_community.vectorstores import FAISS

        print("embedding_function---", embedding_function)
        print("docs---", docs)
        faiss_db = FAISS.from_documents(docs, embedding_function)
        persist_directory ="/temp/faiss_db/"+'ajaz'

        if os.path.exists(persist_directory):

            local_db = FAISS.load_local(persist_directory, embedding_function, allow_dangerous_deserialization=True)

            local_db.merge_from(faiss_db)

            local_db.save_local(persist_directory)

        else:

            faiss_db.save_local(persist_directory)

        return "Completed"

    except Exception as e:

        logger.info('Exception in FaissDB', e)

        raise e

loader_data = loader(file_path,metadata)
embeddings_data = embeddings()

faiss_data_store = FaissDB(embeddings_data,loader_data)

print('File saved to faiss db successfully')

######

def get_similar_docs_faiss_db(embeddings: Any = None, persist_directory: str = "", query: str = '',
                              k: int = None):
    try:

        from langchain_community.vectorstores import FAISS

        db = FAISS.load_local(persist_directory, embeddings, allow_dangerous_deserialization=True)

        docs = db.similarity_search(query, k=k)

        return docs

    except Exception as e:

        logger.info('Exception in get_similar_docs_faiss_db', e)

        raise e

from langchain_community.llms import Ollama


llm = Ollama(
    model="qwen2.5:3b",
    temperature=0.7,
    top_p=0.9
)



def generate_response(
    question,
    context,
    prompt_template
):
    prompt_parts = []

    if context:
        prompt_template = prompt_template.replace("{{context}}", str(context))

    prompt_template = prompt_template.replace("{{question}}", str(question))
    prompt_parts.append(prompt_template)
    prompt_parts.append("Answer:")

    prompt = "\n\n".join(prompt_parts)
    print(">>>>>>>prompt>>>>>>>>>>>", prompt)

    response = llm.invoke(prompt)

    return response.strip()


similar_documents = get_similar_docs_faiss_db(embeddings=embeddings_data,
                                              persist_directory="/temp/faiss_db/" + 'ajaz',
                                              query="List down the skills mentioned in the resume",
                                              k=3)
prompt_template="Use the following pieces of context {{context}} to provide a concise answer to the question to which steps need to be fetched. If context does not have the answer, just give response \'I\'m sorry, I don\'t have enough information to provide an answer to the question.\'. Question: {{question}}"

answer = generate_response("List down the skills mentioned in the resume", similar_documents,prompt_template)

print('Answer:', answer)