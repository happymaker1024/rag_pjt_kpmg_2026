from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
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
def describe_dish_flavor(query):
    # pass
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
            Persona: You are a highly skilled and perceptive culinary expert with a deep understanding of flavors, aromas, and textures in a wide variety of cuisines. Your personality is professional, insightful, and approachable, dedicated to helping users understand and appreciate the complexities of taste in their food experiences. You are passionate about exploring subtle nuances in ingredients and dishes, making flavor analysis accessible and engaging.

            Role: Your role is to guide users in understanding and analyzing the taste, aroma, texture, and overall flavor profile of various foods. You provide detailed descriptions of flavors and offer insights into how different ingredients, cooking techniques, and seasonings influence the dish's final taste. You also help users make informed choices about ingredient combinations and cooking methods to achieve desired flavors in their culinary creations.
            Examples:
            Flavor Profile Analysis: If a user describes a dish with grilled lamb seasoned with rosemary and garlic, you might explain how the earthy, woody notes of rosemary enhance the rich, savory flavor of the lamb. You could also describe how the caramelization from grilling adds a layer of smokiness, balanced by the mild sweetness of roasted garlic.
            Texture and Mouthfeel Explanation: If a user is tasting a creamy mushroom risotto, you might highlight the importance of the dish’s creamy, velvety texture achieved through the slow release of starch from Arborio rice. You could also mention how the umami-rich flavor of mushrooms adds depth to the dish, while the cheese provides a slight saltiness that balances the creaminess.
            Pairing Suggestions: If a user is preparing a spicy Thai green curry, you could recommend balancing its heat with a slightly sweet or acidic side, such as a cucumber salad or coconut rice. You might explain how the coolness of cucumber contrasts with the curry’s heat, and how the subtle sweetness in coconut rice tempers the dish’s spiciness, creating a harmonious dining experience.
         """),
        ("human", """
            이미지의 요리명과 풍미를 한 문장으로 요약해 주세요. 영어로 표현해 주세요.
        """)
    ])

    # image url list
    template = []

    if query.get("image_urls"):
        template += [{"image_url": image_url} for image_url in query["image_urls"]]

    prompt += HumanMessagePromptTemplate.from_template(template)

    llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=4095)

    chain = prompt | llm | StrOutputParser()

    return chain

# 음식에 맛는 와인 top-5 검색
def search_wines(dish_flavor):
    embedding = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL"))
    vector_store = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        embedding=embedding,
        namespace=os.getenv("PINECONE_NAMESPACE")
    )

    results =  vector_store.similarity_search(dish_flavor, namespace=os.getenv("PINECONE_NAMESPACE"), k=5)

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
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=4095)
    # chain = prompt | llm | StrOutputParser()
    chain = prompt | llm | JsonOutputParser()

    return chain

# wine pairing 함수
def wine_pairing(img_urls):
    r1 = RunnableLambda(describe_dish_flavor)
    r2 = RunnableLambda(search_wines)
    r3 = RunnableLambda(recommand_wines)

    chain = r1 | r2 | r3
    res = chain.invoke({
        "image_urls": [img_urls]
    })
    # 최종결과 return
    return res

if __name__ == "__main__":

    img_url = input("image url을 입력하세요. >> ")
    # img_url = 'https://sitem.ssgcdn.com/95/55/96/item/1000346965595_i1_750.jpg'
    # img_url = 'https://www.sbfoods-worldwide.com/ko/recipes/deq4os00000008l9-img/10_Stake_A.jpg'
    print(wine_pairing(img_url))