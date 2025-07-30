#pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring    
'''Sistema de Recuperação de Informação com LangChain e OpenAI'''
import os
from dotenv import load_dotenv

# LLM e embeddings da OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Separação de texto e armazenamento vetorial
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

# Prompt templating
from langchain_core.prompts import ChatPromptTemplate

# Criação das chains modernas
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# Representação de documentos
from langchain.schema import Document

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("A chave da API do OpenAI não foi encontrada. Verifique o arquivo .env.")

class SistemaRag:
    """Sistema de Recuperação de Informação utilizando LangChain e OpenAI."""
    def __init__(self):
        """Inicializa o sistema RAG com LLM, embeddings e base de conhecimento."""
        self.llm = ChatOpenAI(model="gpt-4.1-mini", api_key=openai_api_key, temperature=0.7)
        self.embedding = OpenAIEmbeddings()

        self.vector_store = None
        self.rag_chain = None

        self.criar_base_conhecimento()
        self.configurar_rag()

    def criar_base_conhecimento(self):
        """Cria a base de conhecimento a partir de documentos e os divide em chunks."""
        documentos = [
            Document(
                page_content=(
                    "A empresa XYZ é uma líder em tecnologia, especializada em soluções de IA e automação."
                ),
                metadata={"source": "manual_rh.pdf", "categoria": "recursos humanos"}
            ),
            Document(
                page_content=(
                    "O reembolso de despesas deve ser solicitado dentro de 30 dias após a compra."
                ),
                metadata={"source": "reembolsos.pdf", "categoria": "recursos humanos"}
            ),
            Document(
                page_content=(
                    "Para suporte técnico, entre em contato com o departamento de TI pelo e-mail."
                ),
                metadata={"source": "suporte_tecnico.pdf", "categoria": "suporte tecnico"}
            ),
            Document(
                page_content=(
                    "Os benefícios incluem plano de saúde, vale-refeição e seguro de vida."
                ),
                metadata={"source": "beneficios.pdf", "categoria": "recursos humanos"}
            )
        ]
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.chunks = splitter.split_documents(documentos)
        print(f"Base de Conhecimento criada: {len(self.chunks)} chunks.")

    def configurar_rag(self):
        """Configura o sistema RAG com armazenamento vetorial e cadeia de recuperação."""
        self.vector_store = Chroma.from_documents(
            self.chunks,
            self.embedding,
            collection_name="conhecimento_empresa",
            persist_directory="vector_store"
        )

        system_prompt = (
            "Você é um assistente inteligente especializado em responder perguntas sobre a empresa XYZ. "
            "Forneça respostas precisas e concisas com base no contexto fornecido."
        )
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "Você é assistente da empresa XYZ. Use o contexto abaixo para responder:"),
            ("system", "Contexto:\n{context}"),
            ("human", "{input}")
        ])
        # cria a cadeia de combinação
        stuff_chain = create_stuff_documents_chain(
        llm=self.llm,
        prompt=prompt_template,
        document_variable_name="context"  # padrão, mas bom declarar explicitamente
        )
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        # cria a cadeia de RAG moderna
        self.rag_chain = create_retrieval_chain(
            retriever,
            stuff_chain
        )

        print("RAG configurado com sucesso 🔧")

    def fazer_pergunta(self, pergunta: str):
        """Realiza uma pergunta ao sistema RAG e retorna a resposta e fontes."""
        if not self.rag_chain:
            raise ValueError("RAG não está configurado.")
        result = self.rag_chain.invoke({"input": pergunta})
        resposta = result.get("answer") or result.get("output") or ""
        fonte_docs = result.get("context", [])
        fontes = [
            {"source": doc.metadata.get("source"), "categoria": doc.metadata.get("categoria", "desconhecida")}
            for doc in fonte_docs
        ]
        return {"resposta": resposta, "fontes": fontes}

if __name__ == "__main__":
    sistema = SistemaRag()
    while True:
        pergunta_usuario = input("Pergunte (ou 'sair'): ")
        if pergunta_usuario.lower() == "sair":
            break
        resp = sistema.fazer_pergunta(pergunta_usuario)
        print("Resposta:", resp["resposta"])
        print("Fontes:")
        for f in resp["fontes"]:
            print(f"- {f['source']} (Categoria: {f['categoria']})")
