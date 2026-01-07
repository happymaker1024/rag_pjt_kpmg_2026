from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from dotenv import load_dotenv
import os

load_dotenv("../.env", override=True)

# LLM을 통한 요리 이미지 -> 맛 풍미
# image 에서 맛(풍미)에 대한 설명 text 생성
from langchain_core.messages import HumanMessage, SystemMessage

def describe_dish_flavor(query):
    """
    query = {
        "image_base64": "..."
    }
    """

    messages = [
        SystemMessage(content="""
            You are a highly skilled culinary expert.
            Identify the dish and summarize its flavor profile in one concise English sentence.
            """),
        HumanMessage(
            content=[
                {"type": "text", "text": "Analyze the dish shown in the image."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{query['image_base64']}"
                    }
                }
            ]
        )
    ]

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=300
    )

    response = llm.invoke(messages)

    return response.content

# 음식에 맛는 와인 top-5 검색
def search_wines(dish_flavor):
    embedding = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL"))
    vector_store = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        embedding=embedding,
        namespace=os.getenv("PINECONE_NAMESPACE")
    )

    results =  vector_store.similarity_search(dish_flavor, k=5)

    return {
        "dish_flavor": dish_flavor,
        "wine_reviews": "\n".join([doc.page_content for doc in results])
    }

# 와인 추천
def recommand_wines(query):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
            Persona: You are a refined and approachable virtual wine sommelier with a deep passion for wines, dedicated to helping users explore and enjoy the world of wine with confidence. Your personality is warm, insightful, and patient, ensuring that users feel at ease while learning about wine, regardless of their experience level.
            Role: Your role is to guide users in selecting wines, pairing them with food, and understanding wine characteristics. You are adept at explaining complex wine concepts such as tannins, acidity, and terroir in a way that is accessible to everyone. In addition, you provide suggestions based on the user’s preferences, budget, and the occasion, helping them find the perfect wine to enhance their dining experience.
            Examples:
            Wine Pairing Recommendation: If a user is preparing a buttery garlic shrimp dish, you might suggest a crisp, mineral-driven Chablis or a New Zealand Sauvignon Blanc, explaining how these wines’ acidity and minerality balance the richness of the butter and complement the flavors of the shrimp.
            Wine Selection for a Casual Gathering: If a user is hosting a casual gathering and needs an affordable, crowd-pleasing wine, you might recommend a fruit-forward Pinot Noir or a light Italian Pinot Grigio. Highlight the wines' versatility and how they pair well with a variety of foods, making them ideal for social settings.
            Wine Terminology Explanation: If a user asks what “terroir” means, you would explain it as the unique combination of soil, climate, and landscape in a wine-growing region that influences the wine's flavor, making each wine distinctive to its origin.
            """
         ),
        ("human", """
            와인 페어링 추천에 아래의 요리와 풍미, 와인 리뷰만을 참고하여 한글로 답변해 주세요.

            요리와 풍미:
            {dish_flavor}
         
            와인 리뷰:
            {wine_reviews}
         
            답변은 다음과 같은 값으로 json 데이터로 리턴해주세요. 결과를 한글로 번역해 주세요.
            recommend_wine :
            recommend_reason :
        """)
    ])
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=4095)
    # chain = prompt | llm | StrOutputParser()
    chain = prompt | llm | JsonOutputParser()

    return chain.invoke(query)

# wine pairing 함수
def wine_pairing(image_base64: str):
    r1 = RunnableLambda(describe_dish_flavor)
    r2 = RunnableLambda(search_wines)
    r3 = RunnableLambda(recommand_wines)

    chain = r1 | r2 | r3
    res = chain.invoke({
        "image_base64": image_base64
    })
    # 최종결과 return
    return res

if __name__ == "__main__":
    import base64

    # 파일 경로 : ..\prepare\images\eye_catch_sushi.jpg
    image_path = input("이미지 파일 경로를 입력하세요. >> ")

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    print(wine_pairing(image_base64))