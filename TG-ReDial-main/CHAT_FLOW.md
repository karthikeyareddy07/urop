# Chat recommendation flow

```mermaid
flowchart TD
    A[User types a message in chat.html] --> B[POST /api/chat<br/>{message, history, profile, watch_history}]
    B --> C[chat_app validates input and builds context]
    C --> D{Recommender has<br/>recommend_with_context?}
    D -- Yes --> E[ChineseRanker.recommend_with_context]
    D -- No --> F[ChineseRanker.recommend<br/>legacy last 8 turns]
    E --> G[Normalize profile and watch_history]
    G --> H[build_enriched_context_text<br/>full dialogue + Users + Watched]
    H --> I[ALBERT encode<br/>mean pool + L2 norm]
    I --> J[ChineseRecommender.forward<br/>projection + MLP head + cosine times temperature + popularity bias]
    J --> K[topk_score_entropy<br/>top 20]
    K --> L{Entropy above<br/>threshold?}
    L -- Yes --> M[Return clarification question<br/>QGEN_INTEREST / HISTORY / GENRE]
    L -- No --> N[Format top k movie recommendations]
    N --> O[chat_app asks LLM to rerank top 5<br/>when no clarification is needed]
    M --> P[JSON response<br/>reply, clarification, preferences, entropy, margin]
    O --> Q[JSON response<br/>reply, recommendations, confidence, preferences]
    P --> R[chat.html renders reply bubble and cards]
    Q --> R

    subgraph Offline_Training[Offline training]
        T1[train.jsonl] --> T2[Precompute ALBERT context embeddings]
        T2 --> T3[Build candidate sets<br/>1 positive + 64 negatives]
        T3 --> T4[Train probe with cross entropy]
        T4 --> T5[Validate valid.jsonl<br/>Hit@K / NDCG@K]
        T5 --> T6[Save v2 checkpoint]
        T6 --> T7[ChineseRanker.load<br/>+ catalog + title embeddings]
    end
    T7 -. loaded recommender .-> E

    classDef input fill:#e8f1ff,stroke:#4a78c2,color:#14233b
    classDef model fill:#eee8ff,stroke:#7654b5,color:#21163a
    classDef decision fill:#fff3d6,stroke:#c89022,color:#3d2b08
    classDef response fill:#e6f6eb,stroke:#458b56,color:#173421
    classDef offline fill:#f2f3f5,stroke:#727985,color:#20242b

    class A,B,C input
    class E,F,G,H,I,J,K model
    class D,L decision
    class M,N,O,P,Q,R response
    class T1,T2,T3,T4,T5,T6,T7 offline
```
