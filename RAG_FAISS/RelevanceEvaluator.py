from langchain.schema import HumanMessage, SystemMessage
from langchain.chat_models.gigachat import GigaChat

class RelevanceEvaluator:
    def __init__(self, llm_credentials, llm_scope, llm_model):
        self.llm = GigaChat(
            credentials=llm_credentials, 
            scope=llm_scope, 
            model=llm_model, 
            verify_ssl_certs=False
        )
    
    def determine_relevance(self, query, retrieved_chunks):
        relevant_chunks = []
        for chunk in retrieved_chunks:
            text = chunk.page_content
            messages = [
                SystemMessage(content='''Ты умный и внимательный бот, который помогает пользователю находить релевантные 
                              ответы на его запросы, используя глубокое понимание контекста.Твоя цель — точно оценить, 
                              насколько данный текст соответствует запросу пользователя.'''),
                HumanMessage(content=f"Является ли следующий Текст: '{text}' релевантным для Запроса: '{query}'? Ответь только Да или Нет.")
            ]
            response = self.llm(messages)
            if "Да" in response.content:
                relevant_chunks.append(1)
            else:
                relevant_chunks.append(0)
        return relevant_chunks
    
    def evaluate_metrics(self, relevant_chunks, k):
        relevant = sum(relevant_chunks)
        Precision_top_k = relevant / k

        score = 0.0
        num_hits = 0.0
        total_relevant_items = relevant

        for i, rel in enumerate(relevant_chunks):
            if rel == 1:
                num_hits += 1.0
                precision_at_i = num_hits / (i + 1.0)
                score += precision_at_i

        if total_relevant_items == 0:
            return 'Релевантные ответы отсутствуют'
        
        print(f"Average Precision at {k}: {score / total_relevant_items}")
        print(f'Precision at {k}: {Precision_top_k}')