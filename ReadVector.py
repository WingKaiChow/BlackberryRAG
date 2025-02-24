import faiss

# Load the index from the binary file
index_path = "faiss_index.bin"  # Replace with your file path
index = faiss.read_index(index_path)

# Get basic information about the index
print(f"Number of vectors in the index: {index.ntotal}")
print(f"Vector dimensionality: {index.d}")
print(f"Index type: {type(index).__name__}")

# Check if the index supports vector reconstruction (not all index types do)
if hasattr(index, "reconstruct"):
    # Reconstruct and print the first vector as an example
    first_vector = index.reconstruct(0)  # Reconstruct vector with ID 0
    print(f"First vector: {first_vector}")
else:
    print("This index type does not support direct vector reconstruction.")