import numpy as np
from typing import List, Dict, Any

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def find_top_k_matches(
    query_vector: List[float], 
    candidate_reports: List[Dict[str, Any]], 
    k: int = 5,
    min_threshold: float = 70.0  # Filter out matches below 70%
) -> List[Dict[str, Any]]:
    scored_results = []
    
    for report in candidate_reports:
        candidate_vector = report.get("embedding")
        if not candidate_vector:
            continue
            
        score = cosine_similarity(query_vector, candidate_vector)
        similarity_percentage = round(score * 100, 2)
        
        # Only keep results that pass the confidence threshold
        if similarity_percentage >= min_threshold:
            match_item = {
                "id": str(report["_id"]),
                "description": report.get("description", ""),
                "image_path": report.get("image_path", ""),
                "similarity_score": similarity_percentage,
                "created_at": report.get("created_at")
            }
            scored_results.append(match_item)
    
    # Sort descending by score
    scored_results.sort(key=lambda x: x["similarity_score"], reverse=True)
    return scored_results[:k]