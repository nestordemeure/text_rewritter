from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

# specify the task
data_folder = "./data"
input_file = "input.txt"
output_file = "output.txt"
rewriting_instructions_file = "prompt.txt"

#----------------------------------------------------------------------------------------
# INPUT

print("\nLOADING:\n")

# read the prompt
with open(f"{data_folder}/{rewriting_instructions_file}", "r+") as rewriting_instructions_file:
    rewriting_instructions = rewriting_instructions_file.read()
    print(f"PROMPT: {rewriting_instructions}")

# read the input
with open(f"{data_folder}/{input_file}", "r+") as input_file:
    input = input_file.read()

# splitting input into chunks
#text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=100, chunk_overlap=0)
chunks = text_splitter.split_text(input)
print(f"Splitted input into {len(chunks)} chunks.")

#----------------------------------------------------------------------------------------
# FIRST REWRITING

print("\nREWRITING:\n")

language_model = OpenAI(max_tokens=500)
output = ""

 # build a model
prompt_template = """You are rewriting a novel, rewrite the full extract according to the instructions:

INSTRUCTIONS:
{instructions}

EXTRACT TO REWRITE:
{chunk}

REWRITTEN EXTRACT:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["instructions", "chunk"])
chain = LLMChain(llm=language_model, prompt=prompt)

# rewrite first chunk
chunk = chunks[0]
rewritten_chunk = chain({'instructions':rewriting_instructions, 'chunk':chunk})['text']
print(f"#!#{rewritten_chunk}")

# save result
output += rewritten_chunk
previous_rewriting = (chunk, rewritten_chunk)

#----------------------------------------------------------------------------------------
# SECOND REWRITING

# build a model
prompt_template = """You are rewriting a novel, rewrite the full extract according to the instructions:

INSTRUCTIONS:
{instructions}

EXTRACT TO REWRITE:
{previous_chunk}
{chunk}

REWRITTEN EXTRACT:
{previous_rewrite}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["instructions", "previous_chunk", "chunk", "previous_rewrite"])
chain = LLMChain(llm=language_model, prompt=prompt)

# rewrite second chunk
chunk = chunks[1]
previous_chunk, previous_rewrite = previous_rewriting
rewritten_chunk = chain({'instructions':rewriting_instructions, 'previous_chunk':previous_chunk, 'chunk':chunk, 'previous_rewrite':previous_rewrite})['text']
print(f"#!#{rewritten_chunk}")

# build a search index for previous rewritings
search_index = FAISS.from_documents([Document(page_content=chunk, metadata={'rewriting':rewritten_chunk})], OpenAIEmbeddings())

# save result
output += rewritten_chunk
previous_rewriting = (chunk, rewritten_chunk)

#----------------------------------------------------------------------------------------
# FURTHER REWRITING

# build a model
prompt_template = """You are rewriting a novel, rewrite the full extract according to the instructions:

INSTRUCTIONS:
{instructions}

EXAMPLES OF REWRITINGS FROM OTHER PARTS OF THE TEXT:

{previous_rewrites}

EXTRACT TO REWRITE:
{previous_chunk}
{chunk}

REWRITTEN EXTRACT:
{previous_rewrite}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["instructions", "previous_rewrites", "previous_chunk", "chunk", "previous_rewrite"])
# builds a model
chain = LLMChain(llm=language_model, prompt=prompt, verbose=False)

# rewrite leftover chunks
for chunk in chunks[2:]:
    # truns the relevant previous rewrites into text
    relevant_rewritings = search_index.similarity_search(chunk, k=3)
    previous_rewrites = ""
    for doc in relevant_rewritings:
        before = doc.page_content
        after = doc.metadata['rewriting']
        previous_rewrites += f"\RAW:\n{before}\nREWRITTEN:\n{after}\n"
    # generate the next chunk
    previous_chunk, previous_rewrite = previous_rewriting
    rewritten_chunk = chain({'instructions':rewriting_instructions, 'previous_rewrites':previous_rewrites, 'previous_chunk':previous_chunk, 'chunk':chunk, 'previous_rewrite':previous_rewrite})['text']
    print(f"#!#{rewritten_chunk}")
    # save result
    search_index.add_texts(texts=[chunk], metadatas=[{'rewriting':rewritten_chunk}])
    output += rewritten_chunk
    previous_rewriting = (chunk, rewritten_chunk)

#----------------------------------------------------------------------------------------
# SAVING

# save the output
with open(f"{data_folder}/{output_file}", "w") as output_file:
    output_file.write(output)
