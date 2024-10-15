import json

# key = 100
# value = 1000
# key_val = {
#     200 : 1500
# }

# print(key_val)

# key_val[key] = value

# print(key_val)

# with open("test.json", "w") as file:
#     json.dump(key_val, file)

with open("test.json", "r") as file:
    key_val = json.load(file)
    print(key_val)
    
key_val[1] = 1
    
with open("test.json", "w") as file:
    json.dump(key_val, file)