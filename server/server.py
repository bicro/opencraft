import os
import sqlite3
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from llama_cpp import Llama, LlamaGrammar

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Database setup
DB_PATH = 'cache.db'

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS word_cache (
            id INTEGER PRIMARY KEY,
            first_word TEXT,
            second_word TEXT,
            result TEXT,
            emoji TEXT
        )
    ''')
    conn.close()

init_db()

# Llama model setup
model_path = os.path.join(os.path.dirname(__file__), "models", "mythomax-l2-kimiko-v2-13b.Q2_K.gguf")
llm = Llama(model_path=model_path, n_ctx=2048, n_threads=4)

# Pydantic models
class WordPair(BaseModel):
    first: str
    second: str

class WordResult(BaseModel):
    result: str
    emoji: str
    is_new_element: bool

# Helper functions
def capitalize_first_letter(string: str) -> str:
    return string.capitalize()

async def craft_new_word_from_cache(first_word: str, second_word: str) -> WordResult | None:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT result, emoji FROM word_cache
        WHERE (first_word = ? AND second_word = ?) OR (first_word = ? AND second_word = ?)
    ''', (first_word, second_word, second_word, first_word))
    result = cursor.fetchone()
    conn.close()
    if result:
        return WordResult(result=result['result'], emoji=result['emoji'], is_new_element=False)
    return None

async def cache_new_word(first_word: str, second_word: str, result: str, emoji: str):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO word_cache (first_word, second_word, result, emoji)
        VALUES (?, ?, ?, ?)
    ''', (first_word, second_word, result, emoji))
    conn.commit()
    conn.close()

WORD_GRAMMAR = LlamaGrammar.from_string(r"""
root ::= "{" ws "\"result\"" ws ":" ws "\"" [a-zA-Z]+ "\"" ws "}"
ws ::= [ \t\n]*
""")

EMOJI_GRAMMAR = LlamaGrammar.from_string(r"""
root ::= "{" ws "\"emoji\"" ws ":" ws "\"" . "\"" ws "}"
ws ::= [ \t\n]*
""")

async def generate_word(first_word: str, second_word: str) -> WordResult:
    system_prompt = (
        f"You are a helpful assistant that helps people to craft new things by combining two words into a new word. "
        f"The most important rules that you have to follow with every single answer that you are not allowed to use the words {first_word} and {second_word} as part of your answer and that you are only allowed to answer with one thing. "
        f"DO NOT INCLUDE THE WORDS {first_word} and {second_word} as part of the answer!!!!! The words {first_word} and {second_word} may NOT be part of the answer. "
        "No sentences, no phrases, no multiple words, no punctuation, no special characters, no numbers, no emojis, no URLs, no code, no commands, no programming. "
        "The answer has to be a noun. "
        "The order of both words does not matter, both are equally important. "
        "The answer has to be related to both words and the context of the words. "
        "The answer can either be a combination of the words or the role of one word in relation to the other. "
        "Answers can be things, materials, people, companies, animals, occupations, food, places, objects, emotions, events, concepts, natural phenomena, body parts, vehicles, sports, clothing, furniture, technology, buildings, technology, instruments, beverages, plants, academic subjects and everything else you can think of that is a noun."
    )
    answer_prompt = f"Reply with the result of what would happen if you combine {first_word} and {second_word}. The answer has to be related to both words and the context of the words and may not contain the words themselves. Respond in JSON format with a 'result' key."
    
    prompt = f"<s>[INST] {system_prompt} {answer_prompt} [/INST]</s>\n"
    result = llm(prompt, max_tokens=100, temperature=0.7, stop=["</s>"], echo=False, grammar=WORD_GRAMMAR)
    
    try:
        word_result = json.loads(result['choices'][0]['text'])['result']
    except (json.JSONDecodeError, KeyError, IndexError):
        raise HTTPException(status_code=500, detail="Unexpected response format from language model")

    emoji_prompt = f"<s>[INST] Reply with one emoji for the word: {word_result}. Respond in JSON format with an 'emoji' key containing a single emoji character. [/INST]</s>\n"
    emoji_result = llm(emoji_prompt, max_tokens=10, temperature=0.7, stop=["</s>"], echo=False, grammar=EMOJI_GRAMMAR)
    
    try:
        emoji = json.loads(emoji_result['choices'][0]['text'])['emoji']
    except (json.JSONDecodeError, KeyError, IndexError):
        emoji = "❓"  # Default emoji if we can't get one

    if (len(word_result.split()) > 3 or
        (first_word.lower() in word_result.lower() and
         second_word.lower() in word_result.lower() and
         len(word_result) < (len(first_word) + len(second_word) + 2))):
        return WordResult(result='', emoji='', is_new_element=False)
    
    return WordResult(result=capitalize_first_letter(word_result), emoji=emoji, is_new_element=True)



async def craft_new_word(first_word: str, second_word: str) -> WordResult:
    cached_result = await craft_new_word_from_cache(first_word, second_word)
    if cached_result:
        return cached_result

    result = await generate_word(first_word, second_word)
    await cache_new_word(first_word, second_word, result.result, result.emoji)
    return result

# Routes
@app.get("/")
async def root():
    return {
        'Water + Fire': await craft_new_word('Water', 'Fire'),
        'Water + Earth': await craft_new_word('Water', 'Earth'),
        'Fire + Earth': await craft_new_word('Fire', 'Earth'),
        'Water + Air': await craft_new_word('Water', 'Air'),
        'Earth + Air': await craft_new_word('Earth', 'Air'),
        'Fire + Air': await craft_new_word('Fire', 'Air')
    }

@app.post("/", response_model=WordResult)
async def create_word(word_pair: WordPair):
    if not word_pair.first or not word_pair.second:
        raise HTTPException(status_code=400, detail="Both words are required")
    
    first_word = capitalize_first_letter(word_pair.first.strip().lower())
    second_word = capitalize_first_letter(word_pair.second.strip().lower())
    
    return await craft_new_word(first_word, second_word)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)