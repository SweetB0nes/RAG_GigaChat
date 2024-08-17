В данном репозитори реализована оценка качества поиска релевантных чанков (пара модулей: эмбеддинг + ретривер) в составе RAG.

Для оценки релевантоности запросов и дальнейших расчетов метрик, был использован GigaChat.


Simple_RAG - BERT для эмбейдингов, cosine_similarity для поиска по 'базе'.

RAG_FAISS - Использование фреймфорка FAISS.

Реализованные метрики:

![My Image](https://habrastorage.org/getpro/habr/upload_files/608/ff8/5ef/608ff85ef451ddf587325e8bfc9b113c.png)
![My Image](https://habrastorage.org/getpro/habr/upload_files/567/6d2/9a5/5676d29a556b8b10fca004167167b40d.png)

