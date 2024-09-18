# DataMergeKit

Initial Setup and Imports:

The script imports numerous libraries for data processing, machine learning, and distributed computing.
Key libraries include PySpark for distributed data processing, PyTorch for deep learning, and various scientific computing libraries like NumPy and SciPy.


Configuration Loading:

The script loads configuration from YAML files, including user settings, dataset configurations, and phrase lists.
It sets up default configurations if they don't exist.


Initialization:

Sets up logging to file and console.
Initializes key variables like chunk size, core count, and processed chunks directory.
Downloads and loads a language detection model.


Data Processing Functions:

Several functions are defined for data processing tasks like filtering conversations, transforming records, and writing chunks to files.


Multithreading Setup:

The script uses Python's multiprocessing library to set up a pool of worker processes.
It creates a queue for input tasks and a shared dictionary for output results.


Main Data Processing Loop:

Iterates through datasets, processing them in chunks.
Each chunk is processed in parallel using the worker pool.
Processing includes filtering, transforming, and optionally rewriting system prompts.


Memory Management in Data Processing:

Data is processed in chunks to manage memory usage.
Processed chunks are immediately written to disk as Parquet files.
Garbage collection is explicitly called to free up memory.


Embedding Generation:

Uses a pre-trained model to generate embeddings for text data.
Embeddings are generated using PyTorch, with GPU acceleration if available.
The text_to_embeddings function is a PySpark UDF (User Defined Function) that allows parallel processing of text data.


Deduplication Process:

Converts Spark DataFrame to Pandas for embedding generation.
Calculates cosine similarity matrix for embeddings.
Uses memory-mapped files and chunked processing to handle large similarity matrices.


Cosine Similarity Calculation:

Implements a chunked approach to calculate cosine similarity.
Uses numpy memmap to handle large matrices that don't fit in memory.
Calculates similarity in sub-chunks to further optimize memory usage.


Memory Management in Similarity Calculation:

Calculates maximum matrix size that can fit in memory.
Uses memory-mapped files to handle matrices larger than available RAM.
Processes data in chunks and sub-chunks based on available memory.


Multithreading in Similarity Calculation:

Uses multiprocessing to parallelize the calculation of similarity chunks.
Each chunk is processed independently, allowing for efficient use of multiple CPU cores.


Deduplication Based on Similarity:

Creates a mask of duplicate entries based on a similarity threshold.
Uses sparse matrices to efficiently handle large similarity matrices.


Final Data Processing:

Removes duplicates from the DataFrame.
Shuffles and samples the data according to specified percentages.


Data Saving:

Saves the processed and deduplicated data to a temporary directory using Spark.
Consolidates the data into a single Parquet file.


Hugging Face Upload:

Provides functionality to upload the processed dataset to Hugging Face Hub.



Key Aspects of Memory Management:

Chunked processing: Data is processed in manageable chunks to avoid overwhelming system memory.
Memory-mapped files: Used for handling large matrices that exceed available RAM.
Sparse matrices: Employed for efficient storage and manipulation of large, sparse similarity matrices.
GPU offloading: Embedding generation is offloaded to GPU if available, freeing up CPU memory.
Explicit garbage collection: Called periodically to free up memory.

Multithreading and Parallelization:

PySpark: Used for distributed processing of large datasets.
Multiprocessing: Employed for parallel processing of data chunks and similarity calculations.
GPU acceleration: Utilized for embedding generation when available.

Offloading:

Disk offloading: Processed chunks are written to disk to free up memory.
GPU offloading: Embedding generation is offloaded to GPU for faster processing.
