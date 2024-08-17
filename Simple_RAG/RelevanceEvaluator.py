from langchain.schema import HumanMessage, SystemMessage

class RelevanceEvaluator:
    def __init__(self, llm):
        self.llm = llm
    
    def determine_relevance(self, query, data_chunks, top_k_indices):
        relevant_chunks = []
        for idx in top_k_indices:
            text = data_chunks[idx]
            messages = [
                SystemMessage(content='''Ты умный и внимательный бот, который помогает пользователю находить релевантные 
                              ответы на его запросы, используя глубокое понимание контекста.Твоя цель — точно оценить, 
                              насколько данный текст соответствует запросу пользователя.'''),
                HumanMessage(content=f"Является ли следующий Текст: '{text}' релевантным для Запроса: '{query}'? Ответь только Да или Нет.")
            ]
            response = self.llm(messages)
            relevant_chunks.append(1 if "Да" in response.content else 0)
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
                score += num_hits / (i + 1.0)

        if total_relevant_items == 0:
            return 'Релевантные ответы отсутствуют'
        
        print(f"Average Precision at {k}: {score / total_relevant_items}")
        print(f'Precision at {k}: {Precision_top_k}')
