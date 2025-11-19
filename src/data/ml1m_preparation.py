
# %%
import tensorflow as tf
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import os

# %%
def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

# %%
def serialize_example(user_id, item_id_list, item_text_list):
    """Convert a user interaction into a serialized tf.train.Example"""
    feature = {
        'user_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[user_id])),
        'item_id': tf.train.Feature(int64_list=tf.train.Int64List(value=item_id_list)),
        'item_text': tf.train.Feature(bytes_list=tf.train.BytesList(value=item_text_list)),

    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def process_and_write(grouped_chunk, file_name):
    """Process a chunk of grouped data and write to a TFRecord file."""
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    with tf.io.TFRecordWriter(file_name, options=options) as writer:
        for index, row in grouped_chunk.iterrows():
            user_id = row['user_id']
            item_id_list = row['item_id']
            item_text_list = row['item_text']
            # Convert item_text_list to bytes
            item_text_list = [text.encode('utf-8') for text in item_text_list]
            serialized_example = serialize_example(user_id, item_id_list, item_text_list)
            writer.write(serialized_example)

def split_and_multiprocess(grouped_data, num_chunks, output_folder_path):
    """Split the grouped data into `num_chunks` and use ProcessPoolExecutor to process each chunk."""
    
    # Split the grouped data into `num_chunks` parts
    chunks = np.array_split(grouped_data, num_chunks)
    output_folder_path = os.path.join(output_folder_path, "all_data")
    create_directory_if_not_exists(output_folder_path)
    # List of process names for output files
    tf_file_names= [os.path.join(output_folder_path, f"part_{i}.tfrecord.gz") for i in range(num_chunks)]
    
    
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=num_chunks) as executor:
        # Submit the tasks to the executor
        futures = [executor.submit(process_and_write, chunk, tf_file_name) 
                   for chunk, tf_file_name in zip(chunks, tf_file_names)]
        
        # Wait for all futures to complete
        for future in futures:
            future.result()  # Ensures any exceptions are raised

    print(f"{num_chunks} TFRecord files created successfully.")

# %%
import pandas as pd
import requests
import zipfile
import io
from typing import Dict

def prepare_movielens_sequential_data_kcore(data_url: str = "https://files.grouplens.org/datasets/movielens/ml-1m.zip") -> Dict[int, int]:
    """
    Downloads the MovieLens-1M dataset, filters movies with fewer than 5 ratings,
    orders the remaining movies by ID, and returns a mapping from movie ID
    to its position in the filtered and ordered list.

    Args:
        data_url: The URL to the MovieLens-1M zip file.

    Returns:
        A dictionary where keys are movie IDs (int) and values are their
        0-indexed positions (int) in the filtered and ID-ordered movie list.
    """
    print(f"Downloading MovieLens-1M dataset from {data_url}...")
    try:
        response = requests.get(data_url)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        print("Download complete. Extracting data...")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading dataset: {e}")
        raise

    # Extract the zip file contents in-memory
    zip_file_object = io.BytesIO(response.content)
    with zipfile.ZipFile(zip_file_object, 'r') as zip_ref:
        # For MovieLens-1M, files are typically 'ml-1m/ratings.dat' and 'ml-1m/movies.dat'
        namelist = zip_ref.namelist()
        
        # Find the specific file paths within the zip
        ratings_path = None
        movies_path = None
        for name in namelist:
            if 'ml-1m/ratings.dat' in name:
                ratings_path = name
            elif 'ml-1m/movies.dat' in name:
                movies_path = name
        
        if not ratings_path or not movies_path:
            raise FileNotFoundError("Could not find 'ratings.dat' or 'movies.dat' within the zip file.")

        # Read ratings.dat (uses '::' separator)
        with zip_ref.open(ratings_path) as f:
            ratings_df = pd.read_csv(f, 
                                     sep='::', 
                                     engine='python', # 'python' engine for multi-character separator
                                     names=['user_id', 'movie_id', 'rating', 'timestamp'])
        print(f"Loaded {len(ratings_df)} ratings.")

        # Read movies.dat (uses '::' separator)
        with zip_ref.open(movies_path) as f:
            movies_df = pd.read_csv(f, 
                                    sep='::', 
                                    engine='python', # 'python' engine for multi-character separator
                                    names=['movie_id', 'title', 'genres'],
                                    encoding='ISO-8859-1')
        print(f"Loaded {len(movies_df)} movies.")

    # --- Data Preprocessing for filtering and mapping ---

    # 1. Create a mapping from movie_id to combined movie text (title + genres)
    movies_df['movie_text'] = movies_df['title'] + " (" + movies_df['genres'] + ")"
    movie_id_to_text = movies_df.set_index('movie_id')['movie_text'].to_dict()

    # 1. Count ratings per movie
    movie_rating_counts = ratings_df['movie_id'].value_counts()
    print(f"Calculated rating counts for {len(movie_rating_counts)} movies.")

    # 2. Filter movies with at least 5 ratings
    movies_to_keep = movie_rating_counts[movie_rating_counts >= 5].index
    filtered_movies_df = movies_df[movies_df['movie_id'].isin(movies_to_keep)].copy()
    print(f"Filtered down to {len(filtered_movies_df)} movies with at least 5 ratings.")

    # 3. Order the remaining movies by their ID
    filtered_movies_df_sorted = filtered_movies_df.sort_values(by='movie_id').reset_index(drop=True)
    print("Remaining movies sorted by ID.")

    # 4. Create a mapping from movie ID to its position in the filtered and ordered list
    movie_id_to_position = {
        movie_id: idx for idx, movie_id in enumerate(filtered_movies_df_sorted['movie_id'])
    }
    print("Created movie ID to position mapping.")
    print(f"Mapping contains {len(movie_id_to_position)} entries.")

    # 5. Group by user and create sequences
    ratings_df_sorted = ratings_df.sort_values(by=['user_id', 'timestamp'])
    user_sequences = []
    # Iterate through groups to build sequences
    for user_id, group in ratings_df_sorted.groupby('user_id'):
        movie_ids = group['movie_id'].tolist()
        movie_ids = [movie_id_to_position[mid] for mid in movie_ids if mid in movie_id_to_position]
        
        # Map movie IDs to their corresponding text descriptions
        # Handle cases where a movie_id from ratings might not be in movies_df (though rare for ML-1M)
        movie_texts = [movie_id_to_text.get(mid, f"Unknown Movie (ID: {mid})") for mid in movie_ids if mid in movie_id_to_position]
        
        user_sequences.append({
            'user_id': user_id,
            'item_id': movie_ids,
            'item_text': movie_texts
        })
    print(f"Generated sequences for {len(user_sequences)} users.")

    # 6. Create the final DataFrame
    final_df = pd.DataFrame(user_sequences)
    print("Final DataFrame created.")
    print(f"DataFrame shape: {final_df.shape}")
    print("Sample data:")
    print(final_df.head())

    return final_df

# %%
df = prepare_movielens_sequential_data_kcore()
split_and_multiprocess(df, num_chunks=16, output_folder_path='data/ml1m')


