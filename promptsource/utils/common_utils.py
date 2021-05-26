
def removeHyphen(example):
    example_clean = {}
    for key in example.keys():
        if "-" in key:
           new_key = key.replace("-","_")
           example_clean[new_key] = example[key]
        else:
           example_clean[key] = example[key]
    example = example_clean
    return example
    
