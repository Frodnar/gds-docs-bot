import streamlit as st
import os
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain_openai import OpenAIEmbeddings
import openai
from langchain_openai import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph

st.title("GDS Docs Bot")

os.environ['OPENAI_API_KEY'] = st.secrets['api_keys']['openai']

url=st.secrets['neo4j_info']['uri']
username=st.secrets['neo4j_info']['user']
password=st.secrets['neo4j_info']['password']

existing_index_return = Neo4jVector.from_existing_index(
    OpenAIEmbeddings(),
    url=url,
    username=username,
    password=password,
    index_name="document_index",
    text_node_property="text",  # Need to define if it is not default
)

graph = Neo4jGraph(
    url=url, username=username, password=password
)

chain = GraphCypherQAChain.from_llm(
    ChatOpenAI(temperature=0), graph=graph, verbose=True, allow_dangerous_requests=True
)

client = openai.OpenAI()

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask a question about GDS."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

        graph_result = chain.invoke(query)

        vector_result = existing_index_return.similarity_search(query, k=1)[0]
        
        final_prompt = f"""You are a helpful question-answering agent. Your task is to analyze
        and synthesize information from two sources: the top result from a similarity search
        (unstructured information) and relevant data from a graph database (structured information).
        Given the user's query: {query}, provide a meaningful and efficient answer based
        on the insights derived from the following data:

        Unstructured information: {vector_result}.
        Structured information: {graph_result} """

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": final_prompt}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})

