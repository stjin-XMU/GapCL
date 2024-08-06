

def onek_encoding_unk(value, choices):
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    if index == -1:
        print(value,choices)
    encoding[index] = 1

    return encoding