def characters(keep_punctuation=True, keep_upper=True, keep_numbers=True) :

    file = open("SherlockHolmes.txt", "r").read()
    alphabet = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ")
    numbers = list("0123456789")

    if keep_upper == False : file = file.lower()

    input_text = list(file)

    if keep_punctuation == False :
        input_text = [i for i in input_text if i in alphabet]

    if keep_numbers == False :
        input_text = [i for i in input_text if i in alphabet]

    chars = list(set(file))
    
    return(input_text, chars)
