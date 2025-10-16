from ddgs import DDGS

# Define your search query
query = "gaza israel news"
dd = DDGS()
# Perform the search
results = DDGS().text(query, max_results=5)
print(results)

# Print the results
for result in results:
    for key, value in result.items():
        print(f"{key.upper()}: {value}\n")

