import streamlit as st
import os
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_openai import OpenAIEmbeddings
import openai
from langchain_openai import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts.prompt import PromptTemplate


st.title("GDS Docs Bot")

st.markdown("""Ask questions about Neo4j's Graph Data Science library via this bot!""")

st.markdown("""It is backed by an AuraDB instance where I've loaded the [documentation](https://neo4j.com/docs/graph-data-science/2.12/) for the most recent full GDS release.  The page data was cloned from the [Github repo](https://github.com/neo4j/graph-data-science/tree/master/doc/modules/ROOT/pages).  The data were then [transformed into a knowledge graph](https://python.langchain.com/docs/how_to/graph_constructing/) as well as [embedded using OpenAI models](https://neo4j.com/developer-blog/neo4j-langchain-vector-index-implementation/) with both Cypher generation and embedding modes being used at query time.""")

st.markdown("""Try asking questions such as:
- *What algorithms are related to breadth first search?*
- *What metrics can node similarity be based on?*
- *What can you tell me about the Leiden algorithm?*""")

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

CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.
Always search for IDs using toLower() and CONTAINS.
Never specify a target node label.
If you're asked a very generic question like 'what do you know about X' or 'tell me about Y' then return all relationships to and from that node.
If you're asked to compare two algorithms, return all relationships to both nodes.
Use the spelled out name for acronyms like BFS (Breadth First Search) and DFS (Depth First Search).
For algorithms that sometimes have hyphens in the names (like K-1 coloring) search both with and without the hyphen.
For algorithms that sometimes end in 'ing' (like triangle counting), search without the 'ing' (triangle count).
Examples: Here are a few examples of generated Cypher statements for particular questions:
# What does the Degree Centrality algorithm measure?
MATCH (a:Algorithm) WHERE toLower(a.id) CONTAINS "degree centrality" MATCH (a)-[:MEASURES]->(m) RETURN m
# What algorithms are related to breadth first search?
MATCH (a:Algorithm) WHERE toLower(a.id) CONTAINS "breadth first search" MATCH (a)-[:RELATED_TO]->(r) RETURN r
# What is the output of the K-nearest neighbors algorithm?
MATCH (a:Algorithm) WHERE toLower(a.id) CONTAINS "k-nearest neighbors" MATCH (a)-[:OUTPUT]->(o) RETURN o
# What metrics can node similarity be based on?
MATCH (a:Algorithm) WHERE toLower(a.id) CONTAINS "node similarity" MATCH (a)-[:BASED_ON]->(m) RETURN m
# What editions does the Graph Data Science library have?
MATCH (l:Library) WHERE toLower(l.id) CONTAINS "graph data science" MATCH (l)-[:HAS_EDITION]->(e) RETURN e
# What do you know about the Leiden algorithm?
MATCH (a:Algorithm) WHERE toLower(a.id) CONTAINS "leiden" MATCH path=(a)-[]-() RETURN path
# What are the differences between BFS and DFS?
MATCH (a:Algorithm) WHERE toLower(a.id) IN ["breadth first search", "depth first search"] MATCH path=(a)-[]-() RETURN path
# What are the pros and cons of the K-1 Coloring algorithm? What about triangle counting?
MATCH (a:Algorithm) WHERE toLower(a.id) IN ["k-1 color", "k1 color", "triangle count", "triangle-count"] MATCH path=(a)-[]-() RETURN path


The question is:
{question}"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

chain = GraphCypherQAChain.from_llm(
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True,
    cypher_llm=ChatOpenAI(temperature=0, model="gpt-4o-mini"),
    qa_llm=ChatOpenAI(temperature=0, model="gpt-4o"),
    cypher_prompt=CYPHER_GENERATION_PROMPT,
    #return_intermediate_steps=True,
)

client = openai.OpenAI()

# if "openai_model" not in st.session_state:
#     st.session_state["openai_model"] = "gpt-4o-mini"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask a question about GDS."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
        
        graph_result = chain.invoke({'query': query})

        vector_result = existing_index_return.similarity_search(query, k=5)
        
        final_prompt = f"""You are a helpful question-answering agent. Your task is to analyze
        and synthesize information from two sources: the top result from a similarity search
        (unstructured information) and relevant data from a graph database (structured information).
        Given the user's query: {query}, provide a meaningful and efficient answer based
        on the insights derived from the following data:

        Unstructured information: {vector_result}.
        Structured information: {graph_result} """

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": m["role"], "content": final_prompt}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        st.markdown("*Returned vector data:*")
        vector_response = st.markdown("> " + vector_result[0].page_content[:300].replace("\n", "\n> ") + "...")
        st.markdown("*Returned KG data:*" )
        #st.markdown("> " + graph_result['intermediate_steps'].replace("\n", "\n> "))
        try:
            graph_response = st.markdown("> " + graph_result['result'].replace("\n", "\n> "))
        except:
            graph_response = st.markdown("> " + graph_result[:300].replace("\n", "\n> "))
        st.text("")
        st.markdown("***Final response:***")
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})

