def process_docs(dataset):
    def normalize(document):
        answer = document["answer_not_matching_behavior"]
        if isinstance(answer, list):
            return {"answer_not_matching_behavior": answer[0]}
        return {}

    return dataset.map(normalize)
