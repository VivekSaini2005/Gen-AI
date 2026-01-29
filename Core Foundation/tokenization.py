import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")

text = "Hey There! My name is Vivek Saini"
tokens = enc.encode(text)

print("Tokens", tokens)
# Tokens [25216, 3274, 0, 3673, 1308, 382, 118495, 74, 336, 61954]

decoded = enc.decode([25216, 3274, 0, 3673, 1308, 382, 118495, 74, 336, 61954])
print("Decoded", decoded)