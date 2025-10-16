import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.decomposition import PCA
import gc
from tqdm import tqdm

def emb2vec(df, columns_to_process, variance_ratio=0.9):
    print("loading model...")
    model = SentenceTransformer(
        'paraphrase-multilingual-MiniLM-L12-v2',
        device='mps'
    )
    print("complete!")
    all_embeddings = {}
    for column in columns_to_process:
        print(f"encoding {column}...")
        texts = df[column].fillna('').astype(str).tolist()
        embeddings = model.encode(
            texts,
            batch_size=64,  # 128や256など、メモリが許す限り大きくする
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False
        )
        all_embeddings[column] = embeddings
        del texts, embeddings
        gc.collect()
    print("building final dataframe...")
    vec_dfs = [df]
    for column, embeddings in all_embeddings.items():
        print(f"applying PCA to {column} embeddings (target variance ratio: {variance_ratio})...")
        pca = PCA(n_components=variance_ratio)
        reduced_embeddings = pca.fit_transform(embeddings)
        n_components = reduced_embeddings.shape[1]
        actual_variance = pca.explained_variance_ratio_.sum()
        print(f"reduced to {n_components} components (explained variance: {actual_variance:.4f})")

        vec_df = pd.DataFrame(
            reduced_embeddings,
            columns=[f'{column}_pca_{j}' for j in range(n_components)],
            index=df.index,
            dtype='float32'
        )
        vec_dfs.append(vec_df)
        del embeddings, reduced_embeddings, pca
        gc.collect()
    df = pd.concat(vec_dfs, axis=1)
    del vec_dfs
    gc.collect()
    print("all process complete!")
    return df