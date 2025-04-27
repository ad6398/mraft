def precision_recall_at_k(retrieved_docs, ground_truth_docs, k):
    """
    Compute average precision@k and recall@k for a batch of queries.

    Args:
        retrieved_docs: List of lists, each list has retrieved doc IDs per query.
        ground_truth_docs: List of sets or lists, each has true relevant doc IDs.
        k: Integer cutoff for top-k.

    Returns:
        avg_precision: Float, average precision@k over all queries.
        avg_recall: Float, average recall@k over all queries.
    """
    precisions = []
    recalls = []
    if type(retrieved_docs[0]) != list:
        retrieved_docs = [retrieved_docs]
    if type(ground_truth_docs[0]) != list:
        ground_truth_docs= [ground_truth_docs]
    for retrieved, relevant in zip(retrieved_docs, ground_truth_docs):
        relevant_set = set(relevant)
        top_k = retrieved[:k]
        true_positives = relevant_set.intersection(top_k)
        
        precision = len(true_positives) / k
        recall = len(true_positives) / len(relevant_set) if relevant_set else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
    
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    
    return avg_precision, avg_recall



