from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

class AllMpnetBaseV2:
    def __init__(self):
        self.sentences = ["This is a sentence", "This is another sentence"]
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")

    def calculate_cosine_similarity_between_sentences(self, input_sentences=None):
        if input_sentences is None:
            input_sentences = self.sentences
        
        if not isinstance(input_sentences, list) or len(input_sentences) != 2:
            raise ValueError("Exactly 2 sentences are required for comparison.")
        
        encoded_embeddings = self.__get_normalized_sentence_embeddings(input_sentences)
        embeddings_prepared_for_similarity = [embedding.view(1, -1) for embedding in encoded_embeddings]
        cosine_similarity_score = F.cosine_similarity(embeddings_prepared_for_similarity[0], embeddings_prepared_for_similarity[1])
        return cosine_similarity_score.item()

    def find_highest_cosine_similarity_pair(self, input_sentences):
        embeddings = self.__encode_and_normalize_sentences(input_sentences)
        highest_similarity_score = float('-inf')
        most_similar_sentence_pair = (None, None)

        total_sentences = len(input_sentences)
        for i in range(total_sentences):
            for j in range(i + 1, total_sentences):
                current_similarity = F.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)).item()

                if current_similarity > highest_similarity_score:
                    highest_similarity_score = current_similarity
                    most_similar_sentence_pair = (input_sentences[i], input_sentences[j])

        return highest_similarity_score, most_similar_sentence_pair

    def process_and_normalize_sentence_embedding(self, input_sentence):
        tokenized_input = self.tokenizer(input_sentence, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            output_from_model = self.model(**tokenized_input)

        pooled_embedding = self.__apply_mean_pooling_to_embedding(output_from_model, tokenized_input['attention_mask'])
        normalized_embedding = F.normalize(pooled_embedding, p=2, dim=1)
        return normalized_embedding.view(-1).tolist()

    def __apply_mean_pooling_to_embedding(self, output_from_model, attention_mask):
        embeddings_of_tokens = output_from_model.last_hidden_state
        expanded_attention_mask = attention_mask.unsqueeze(-1).expand_as(embeddings_of_tokens).float()
        weighted_sum_embeddings = torch.sum(embeddings_of_tokens * expanded_attention_mask, axis=1)
        sum_mask = torch.clamp(expanded_attention_mask.sum(1), min=1e-9)
        mean_pooled_embeddings = weighted_sum_embeddings / sum_mask
        return mean_pooled_embeddings

    def __encode_and_normalize_sentences(self, sentences):
        tokenized_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            model_output = self.model(**tokenized_input)

        pooled_embeddings = self.__apply_mean_pooling_to_embedding(model_output, tokenized_input['attention_mask'])
        normalized_embeddings = F.normalize(pooled_embeddings, p=2, dim=1)
        return normalized_embeddings

    def __get_normalized_sentence_embeddings(self, sentences):
        # Tokenize sentences
        tokenized_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**tokenized_input)

        # Apply mean pooling (assuming you have a method like __apply_mean_pooling_to_embedding)
        embeddings = self.__apply_mean_pooling_to_embedding(outputs, tokenized_input['attention_mask'])

        # Normalize embeddings
        normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return normalized_embeddings

# Now instantiate the class
model_revised = AllMpnetBaseV2()
