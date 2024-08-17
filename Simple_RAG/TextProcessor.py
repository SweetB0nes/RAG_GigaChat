class TextProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.lines = self._load_file()
    
    def _load_file(self):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        return [line.replace('\xa0', ' ').strip() for line in lines]
    
    def split_into_chunks(self, text, chunk_size):
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                while end > start and text[end] not in [' ', '\n', '.', ',', ';', '!', '?']:
                    end -= 1
            chunks.append(text[start:end].strip())
            start = end
        return chunks
    
    def get_chunks(self, chunk_size):
        return self.split_into_chunks(" ".join(self.lines), chunk_size)
