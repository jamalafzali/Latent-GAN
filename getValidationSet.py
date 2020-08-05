# Splitting data between train and validation sets
ints = list(range(time_steps))
random.shuffle(ints)
int_to_split = int(val_percent * time_steps)
train_ints = ints[int_to_split:]
val_ints = ints[:int_to_split]

print()